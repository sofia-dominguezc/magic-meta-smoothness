import math
from tqdm import tqdm
from functools import partial
from contextlib import contextmanager
from typing import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD
from torch.func import grad_and_value, functional_call

from src.data import Loader
from src.logging import Logger, op_norm
from src.eigvals import torch_lobpcg
from src.models import OptimState, dict_zip


LossFn = Callable[[Tensor, Tensor], Tensor]  # (model output [B, C], int labels [B]) -> per-example losses [B]
VJP = Callable[[dict[str, Tensor]], dict[str, Tensor]]  # of a gradient: v -> (H v,), from torch.func.vjp


def cross_entropy(out: Tensor, Y: Tensor) -> Tensor:
    return F.cross_entropy(out, Y, reduction="none")


def mse(out: Tensor, Y: Tensor) -> Tensor:
    """Squared error to the one-hot labels"""
    return 0.5 * (out - F.one_hot(Y, out.shape[1]).to(out.dtype)).pow(2).sum(-1)


LOSSES: dict[str, LossFn] = {"ce": cross_entropy, "mse": mse}


def metrics(out: Tensor, Y: Tensor) -> dict[str, Tensor]:
    """Per-example classification metrics of the model: accuracy and margin."""
    correct = out.gather(1, Y[:, None])[:, 0]
    best_other = out.scatter(1, Y[:, None], float("-inf")).max(1).values
    return {"acc": (out.argmax(1) == Y).to(out.dtype), "margin": correct - best_other}


def model_call(model: nn.Module, params: dict[str, Tensor], X: Tensor, *, bf16: bool = False) -> Tensor:
    """`model(params, X)`, with the matmuls in bfloat16 (autocast) if `bf16`"""
    with torch.autocast(X.device.type, torch.bfloat16, enabled=bf16):
        out: Tensor = functional_call(model, params, (X,))
    return out.to(X.dtype)


def eval_loss(
    model: nn.Module, loss_fn: LossFn, params: dict[str, Tensor], loader: Loader, indices: Tensor,
    *, grad: bool = False, bf16: bool = False,
) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    """Mean loss of `params` over the examples `indices` of `loader`, accumulated batch by batch to bound the memory.
    Returns: (loss, {metric: mean value}, {param: d loss / d param} if `grad` else {})."""
    n = len(indices)

    def batch_loss(params: dict[str, Tensor], X: Tensor, Y: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """This batch's share of the mean loss, and of the mean metrics."""
        out = model_call(model, params, X, bf16=bf16)
        return loss_fn(out, Y).sum() / n, {k: v.sum() / n for k, v in metrics(out, Y).items()}

    params = {k: p.detach() for k, p in params.items()}
    loss, mean_metrics = torch.zeros((), device=loader.device), {}
    grads = {k: torch.zeros_like(p) for k, p in params.items()} if grad else {}
    for X, Y in loader.iter_batches(indices):
        if grad:
            d_params, (term, batch_metrics) = grad_and_value(batch_loss, has_aux=True)(params, X, Y)
            grads = {k: grads[k] + d_params[k] for k in grads}
        else:
            term, batch_metrics = batch_loss(params, X, Y)
        loss += term.detach()
        for k, v in batch_metrics.items():
            mean_metrics[k] = mean_metrics.get(k, 0) + v.detach()
    return loss, mean_metrics, grads


def hessian_eigs(
    vjp_fn: VJP, eigvecs: dict[str, Tensor], precond: Mapping[str, Tensor | float] | None = None, sign: float = 1,
) -> tuple[Tensor, dict[str, Tensor], int]:
    """Top k eigenpairs of sqrt(P) H sqrt(P) by LOBPCG, or the negative ones if `sign` is -1.
    `vjp_fn(v)` computes Hv where v is in dict form, P are the step sizes `precond` (None: 1).
    Returns: (eigenvalues [k], eigenvectors like `eigvecs`, LOBPCG iterations).
    """
    def flat(d: dict[str, Tensor]) -> Tensor:  # {name: [m, *shape]} -> [#params, m]
        return torch.cat([v.flatten(1) for v in d.values()], dim=1).T

    def unflat(V: Tensor) -> dict[str, Tensor]:
        return {
            name: v.T.reshape(-1, *shape)
            for v, (name, shape) in zip(V.split([shape.numel() for shape in shapes.values()]), shapes.items())
        }

    shapes = {name: v.shape[1:] for name, v in eigvecs.items()}
    hvp = torch.func.vmap(vjp_fn)  # batched over the leading dim m
    sqrt_p = None if precond is None else flat({
        name: torch.as_tensor(precond[name], dtype=v.dtype, device=v.device).expand(1, *shape)
        for (name, shape), v in zip(shapes.items(), eigvecs.values())
    }).sqrt()

    def A(V: Tensor) -> Tensor:  # [#params, m] -> [#params, m] (LOBPCG uses m = k and 3k)
        if sqrt_p is not None:
            V = sqrt_p * V
        HV = flat(hvp(unflat(V)))
        return sign * (HV if sqrt_p is None else sqrt_p * HV)

    eigvals, V, n_iter = torch_lobpcg(A, flat(eigvecs))
    return sign * eigvals, unflat(V), n_iter


@dataclass(frozen=True, kw_only=True)
class Magic:
    """Training of `model` on `loader`, REWIND, MAGIC, and logged training.
    Supports bf16 autocast on all methods, and compile on `train`.
    """

    model: nn.Module
    loader: Loader
    loss_fn: LossFn
    bf16: bool = False
    compile: bool = False

    @property
    def device(self) -> torch.device | str | None:
        return self.loader.device

    @partial(torch.func.grad, argnums=1)
    def grad_fn(self, params: dict[str, Tensor], X: Tensor, Y: Tensor, weights: Tensor | None = None) -> Tensor:
        """The gradient of the mean loss on the batch, with the examples weighted by `weights` (None: all 1)"""
        losses = self.loss_fn(model_call(self.model, params, X, bf16=self.bf16), Y)
        return ((1 if weights is None else weights) * losses).mean()

    def train[T: OptimState](
        self, init_state: T, *, n_steps: int, start: int = 0,
        weights: Tensor | None = None, ckpt_every: int | None = None, pbar: bool = True,
    ) -> tuple[T, list[T]]:
        """Train on steps [start, start + n_steps) with the per-example weights `weights`.
        `weights` can be 2D (batched), `init_state` can also be batched (but not both).
        """
        state = init_state.apply(lambda t: t.to(self.device, non_blocking=True, copy=True))
        if weights is None:
            weights = torch.ones(len(self.loader.dataset))
        weights = weights.to(self.device, non_blocking=True)
        grad_fn = self.grad_fn
        name, p = next(iter(state.params.items()))
        if p.ndim > self.model.get_parameter(name).ndim:  # batched state
            grad_fn = torch.func.vmap(grad_fn, in_dims=(0, None, None, None))
        elif weights.ndim == 2:
            state = state.apply(lambda t: t.expand(len(weights), *t.shape).clone())
            grad_fn = torch.func.vmap(grad_fn, in_dims=(0, None, None, 0))
        if self.compile:
            grad_fn = torch.compile(grad_fn)

        ckpts: list[T] = []
        for t in tqdm(range(start, start + n_steps), disable=not pbar, desc="train"):
            if ckpt_every and (t - start) % ckpt_every == 0:
                ckpts.append(state.apply(lambda t: t.to("cpu", copy=True)))
            X, Y, idx = self.loader.get_batch(t)
            grads = grad_fn(state.params, X, Y, weights[..., idx])
            state.step(t, grads)
        return state.apply(lambda t: t.cpu()), ckpts

    def train_logged[T: OptimState](self, init_state: T, *, n_steps: int, start: int = 0, log: "TrainLogger") -> T:
        """Train on all the data on steps [start, start + n_steps) with the logs of `log` (`TrainLogger`)."""
        log.logger.eos = init_state.stability_threshold
        state = init_state.apply(lambda t: t.to(self.device, copy=True))
        for t in tqdm(range(start, start + n_steps)):
            X, Y, _ = self.loader.get_batch(t)
            with log.dual(t, state):  # attach tangent in-place + detach `state` on exit
                params = {name: p.requires_grad_() for name, p in state.params.items()}
                out = model_call(self.model, params, X, bf16=self.bf16)
                loss = self.loss_fn(out, Y).mean()
                grads = torch.autograd.grad(loss, params)
                state.step(t, grads)

            train_metrics = metrics(out.detach(), Y)
            log.logger.log(t, loss=loss, **{k: v.mean() for k, v in train_metrics.items()})
            log.logger.log_norms(t, "params", params)
            log.logger.log_norms(t, "grads", grads)
            if state.lr.kind != "constant" or state.lr.warmup:
                log.logger.log(t, lr=torch.tensor(state.lr(t)))
            if log.n_eigs > 0 and (t - start) % log.eig_every == 0:
                vjp_fn = torch.func.vjp(lambda p: self.grad_fn(p, X, Y), {k: p.detach() for k, p in params.items()})[1]
                log.log_eigs(t, lambda v: vjp_fn(v)[0], state)
            if log.eval_every and ((t - start + 1) % log.eval_every == 0 or t - start == n_steps - 1):
                log.log_eval(t, state.params, self.model, self.loss_fn, bf16=self.bf16, last=t - start == n_steps - 1)
        return state.apply(lambda t: t.cpu())

    def replay[T: OptimState](
        self, init_state: T, *, n_steps: int, start: int = 0, passes: int = 2, pbar: bool = True
    ) -> Iterator[T]:
        """Reconstruct the states [start, start + n_steps] in reverse order
        Cost: `passes` trainings. Memory: `passes * n_steps^(1/passes)` states.
        """
        if n_steps == 0:
            yield init_state
            return
        chunk = math.ceil(n_steps ** ((passes - 1) / passes))
        final, ckpts = self.train(init_state, n_steps=n_steps, start=start, ckpt_every=chunk, pbar=pbar)
        yield final
        for i, ckpt in reversed(list(enumerate(ckpts))):
            r_start = start + i * chunk
            r_steps = min(chunk, start + n_steps - r_start)
            yield from self.replay(
                ckpt, n_steps=r_steps - 1, start=r_start, passes=passes - 1, pbar=False
            )

    def backprop(
        self, init_state: OptimState, *, n_steps: int, start: int = 0,
        test_loader: Loader, eval_sets: dict[str, Tensor], passes: int = 2, logger: Logger | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """MAGIC: backpropagate the functions f = "mean loss over a few examples"
        through steps [start, start + n_steps) from `init_state`. Also log the norms of df/ds_t.

        Returns: ({name: f(s_T)}, {name: df(s_T)/dw}).
        Cost: `passes + 1` trainings + one backward through training.
        Memory: `passes * n_steps^(1/passes) + #fns` states.
        """
        if (n_fns := len(eval_sets)) == 0:
            return {}, {}
        states = self.replay(init_state, n_steps=n_steps, start=start, passes=passes)
        final_state = next(states).apply(lambda t: t.to(self.device))  # 1x training

        outputs: dict[str, Tensor] = {}  # f(s_T)
        betas: dict[str, Tensor] = {}    # df/dw
        delta = final_state.apply(lambda t: t.new_zeros(n_fns, *t.shape))  # df/ds_t
        for i, (name, indices) in enumerate(eval_sets.items()):
            f_value, _, f_grad = eval_loss(
                self.model, self.loss_fn, final_state.params, test_loader, indices, grad=True, bf16=self.bf16
            )
            outputs[name] = f_value
            betas[name] = torch.zeros(len(self.loader.dataset), device=self.device)
            for d, g in dict_zip(delta.params, f_grad).values():
                d[i] = g

        steps = reversed(range(start, start + n_steps))
        for t, state in zip(steps, tqdm(states, total=n_steps, desc="magic")):
            X, Y, indices = self.loader.get_batch(t)
            state = state.apply(lambda t: t.to(self.device, non_blocking=True).requires_grad_())
            weights = torch.ones_like(indices, dtype=X.dtype).requires_grad_()

            out = model_call(self.model, state.params, X, bf16=self.bf16)
            loss = (weights * self.loss_fn(out, Y)).mean()
            grads = torch.autograd.grad(loss, state.params, create_graph=True)
            new_state = replace(state).step(t, grads)

            *d_state, d_weights = torch.autograd.grad(
                list(new_state), [*state, weights], list(delta), is_grads_batched=True
            )
            delta = delta.unflatten(d_state)
            for i, name in enumerate(eval_sets):
                betas[name].index_add_(0, indices, d_weights[i])
                if logger is not None:
                    logger.log_norms(t, f"delta/{name}", delta.apply(lambda t: t[i]))

        return outputs, betas


@dataclass
class TrainLogger:
    """Handles the computationally expensive logs of `Magic.train_logged`.
    - Top pos/neg pre-conditioned Hessian eigenvalues
    - Norm of `ds_t/ds_0 @ v_0` for a random initial direction v_0
    - Validation metrics
    """

    logger: Logger
    test_loader: Loader
    n_top_eigs: int = 1
    n_neg_eigs: int = 0
    eig_every: int = 10
    tangent: bool = True
    eval_every: int = 100
    n_test: int | None = None
    eigvecs: dict[int, dict[str, Tensor]] = field(default_factory=dict)  # sign (1: top, -1: most negative) -> LOBPCG warm start
    tangent_state: OptimState | None = None  # ds_t/ds_0 v, once the training started

    @property
    def n_eigs(self) -> int:
        return self.n_top_eigs + self.n_neg_eigs

    @contextmanager
    def dual(self, t: int, state: OptimState):
        """On enter: attach a forward-mode AD tangent to `state` in-place.
        On exit: read the output tangent from `state` and detach it.
        """
        if self.tangent:
            if self.tangent_state is None:
                g = torch.Generator().manual_seed(t)
                tangent = state.apply(lambda s: torch.randn(s.shape, generator=g, dtype=s.dtype).to(s.device))
                self.tangent_state = tangent.apply(lambda s: s / op_norm(s, n_iter=10)[0])
            fwAD.enter_dual_level()
            tangents = iter(self.tangent_state)
            state.update(lambda s: fwAD.make_dual(s, next(tangents)))
        yield
        if self.tangent:
            def tangent_of(s: Tensor) -> Tensor:
                tangent = fwAD.unpack_dual(s).tangent
                assert tangent is not None
                return tangent.detach()

            self.tangent_state = state.apply(tangent_of)
            fwAD.exit_dual_level()
            self.logger.log_norms(t, "tangent", self.tangent_state)
        state.update(lambda s: s.detach())

    def log_eigs(self, t: int, vjp_fn: VJP, state: OptimState):
        """Log the Hessian eigenvalues (`hessian_eigs`, preconditioned by `state`), the eigenvectors' norms and
        their cosine with the tangent."""
        for sign, k in ((1, self.n_top_eigs), (-1, self.n_neg_eigs)):
            if k == 0:
                continue
            if sign not in self.eigvecs:  # random start
                g = torch.Generator().manual_seed(t)
                self.eigvecs[sign] = {
                    name: torch.randn(k, *p.shape, generator=g, dtype=p.dtype).to(p.device)
                    for name, p in state.params.items()
                }
            eigvals, eigvecs, n_iter = hessian_eigs(vjp_fn, self.eigvecs[sign], state.precond(t), sign)
            self.eigvecs[sign] = eigvecs
            self.logger.log(
                t, **{f"eig/{sign * (i + 1)}": lam for i, lam in enumerate(eigvals.unbind())},
                **{f"lobpcg_iters/{sign}": torch.tensor(n_iter)},
            )
            for i in range(k):
                self.logger.log_norms(t, f"eigval/{sign * (i + 1)}", {name: v[i] for name, v in eigvecs.items()})
            if self.tangent_state is not None:
                tangent = self.tangent_state.params
                dot = sum((s * v[0]).sum() for s, v in dict_zip(tangent, eigvecs).values())
                cos = dot / sum(s.pow(2).sum() for s in tangent.values()) ** 0.5
                self.logger.log(t, **{"tangent_top_eig" if sign > 0 else "tangent_neg_eig": cos})

    def log_eval(self, t: int, params: dict[str, Tensor], model: nn.Module, loss_fn: LossFn, *, bf16: bool = False, last: bool = False):
        """The test loss and metrics of `params`, on random examples seeded by `t` unless `last`."""
        test_ds = self.test_loader.dataset
        indices = torch.arange(len(test_ds)) if self.n_test is None or last else test_ds.subset(self.n_test, index=t)
        loss, mean_metrics, _ = eval_loss(model, loss_fn, params, self.test_loader, indices, bf16=bf16)
        self.logger.log(t, test_loss=loss, **{f"test_{k}": v for k, v in mean_metrics.items()})
