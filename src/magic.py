import math
from tqdm import tqdm
from dataclasses import dataclass, replace
from typing import Callable, Iterator

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD
from torch.func import grad_and_value, functional_call

from src.data import EvalLoader, Loader, Subset
from src.logging import Logger, op_norm
from src.eigvals import top_hessian_eigs
from src.models import OptimState, dict_zip


LossFn = Callable[[Tensor, Tensor], Tensor]  # (model output [B, C], int labels [B]) -> per-example losses [B]


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
    model: nn.Module, loss_fn: LossFn, params: dict[str, Tensor],
    loader: EvalLoader, grad: bool = False, bf16: bool = False,
) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    """Mean loss over the examples of `loader`, accumulated batch by batch to bound the memory.
    Returns: (loss, {metric: mean value}, {param: d loss / d param} if `grad` else {})."""

    def batch_loss(params: dict[str, Tensor], X: Tensor, Y: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """This batch's share of the mean loss, and of the mean metrics."""
        out = model_call(model, params, X, bf16=bf16)
        return (
            loss_fn(out, Y).sum() / len(loader),
            {k: v.sum() / len(loader) for k, v in metrics(out, Y).items()},
        )

    params = {k: p.detach() for k, p in params.items()}
    loss, mean_metrics = torch.zeros((), device=loader.device), {}
    grads = {k: torch.zeros_like(p) for k, p in params.items()} if grad else {}
    for X, Y in loader:
        if grad:
            d_params, (term, batch_metrics) = grad_and_value(batch_loss, has_aux=True)(params, X, Y)
            grads = {k: grads[k] + d_params[k] for k in grads}
        else:
            term, batch_metrics = batch_loss(params, X, Y)
        loss += term.detach()
        for k, v in batch_metrics.items():
            mean_metrics[k] = mean_metrics.get(k, 0) + v.detach()
    return loss, mean_metrics, grads


@dataclass(frozen=True, kw_only=True)
class Magic:
    """Training of `model` on `loader`, and MAGIC
    - train: plain training, supports multiple data weightings
    - train_logged: one training with logs
    - backprop: MAGIC, compute the effect of weights df(s_T)/dw on model training
    `bf16`: the model forward in bfloat16 (autocast)
    """

    model: nn.Module
    loader: Loader
    loss_fn: LossFn
    bf16: bool = False
    compile: bool = False

    logger: Logger | None = None
    n_top_eigs: int = 1
    n_neg_eigs: int = 0
    eig_every: int = 10
    log_tangent: bool = True

    test_loader: EvalLoader | None = None
    eval_every: int = 100  # how often to eval
    n_test: int | None = None  # how many samples to use for eval

    @property
    def device(self) -> torch.device | str | None:
        return self.loader.device

    def train(
        self, init_state: OptimState, *, n_steps: int, start: int = 0,
        weights: Tensor | None = None, ckpt_every: int | None = None,
        pbar: bool = True,
    ) -> tuple[OptimState, list[OptimState]]:
        """Train on steps [start, start + n_steps) with the per-example weights `weights`.
        `weights` can be None (all ones), (#datapoints), or (#n_weights, #datapoints)
        `init_state` can be batched if `weights` are too
        Returns: (final state, states (start, start + ckpt_every, ...))
        """
        @torch.func.grad
        def grad_fn(params: dict[str, Tensor], weights: Tensor, X: Tensor, Y: Tensor) -> Tensor:
            """The training objective of one step with samples weighted by `weights`"""
            out = model_call(self.model, params, X, bf16=self.bf16)
            return (weights * self.loss_fn(out, Y)).mean()

        state = init_state.apply(lambda t: t.to(self.device, non_blocking=True, copy=True))
        if weights is None:
            weights = torch.ones(len(self.loader.dataset))
        weights = weights.to(self.device, non_blocking=True)
        if weights.ndim == 2:
            name, p = next(iter(state.params.items()))
            if p.ndim == self.model.get_parameter(name).ndim:  # not already batched
                state = state.apply(lambda t: t.expand(len(weights), *t.shape).clone())
            grad_fn = torch.func.vmap(grad_fn, in_dims=(0, 0, None, None))
        if self.compile:
            grad_fn = torch.compile(grad_fn)

        ckpts: list[OptimState] = []
        for t in tqdm(range(start, start + n_steps), disable=not pbar, desc="train"):
            if ckpt_every and (t - start) % ckpt_every == 0:
                ckpts.append(state.apply(lambda t: t.to("cpu", copy=True)))
            X, Y, idx = self.loader.get(t)
            grads = grad_fn(state.params, weights[..., idx], X, Y)
            state = state.step(grads, t)
        return state.apply(lambda t: t.cpu()), ckpts

    def train_logged(self, init_state: OptimState, *, n_steps: int, start: int = 0) -> OptimState:
        """Train on all the data on steps [start, start + n_steps) while logging.
        Logs the top Hessian eigenvalues, the norm of ds_t/ds_0 on a random direction, and more.
        """
        assert self.logger is not None
        self.logger.eos = init_state.stability_threshold
        state = init_state.apply(lambda t: t.to(self.device, copy=True))

        eigvecs: dict[int, Tensor] = {}  # sign (1: top, -1: most negative) -> LOBPCG warm start
        tangent: OptimState | None = None  # ds_t/ds_0 @ v
        if self.log_tangent:
            g = torch.Generator().manual_seed(start)
            tangent = state.apply(lambda t: torch.randn(t.shape, generator=g, dtype=t.dtype))
            tangent = tangent.apply(lambda t: (t / op_norm(t, n_iter=10)[0]).to(self.device))

        for t in tqdm(range(start, start + n_steps)):
            X, Y, _ = self.loader.get(t)
            if state.lr.kind != "constant" or state.lr.warmup:
                self.logger.log(t, lr=torch.tensor(state.lr(t)))
            if tangent is not None:
                fwAD.enter_dual_level()
                state = state.unflatten([fwAD.make_dual(s, d) for s, d in zip(state, tangent)])
            params = {name: p.requires_grad_() for name, p in state.params.items()}
            log_eigs = self.n_top_eigs + self.n_neg_eigs > 0 and (t - start) % self.eig_every == 0

            out = model_call(self.model, params, X, bf16=self.bf16)
            loss = self.loss_fn(out, Y).mean()
            grads = torch.autograd.grad(loss, params, create_graph=log_eigs)
            state = state.step(grads, t)

            if tangent is not None:
                tangent = state.apply(lambda s: fwAD.unpack_dual(s).tangent.detach())
                fwAD.exit_dual_level()
            state = state.apply(lambda t: t.detach())

            train_metrics = metrics(out.detach(), Y)
            self.logger.log(t, loss=loss, **{k: v.mean() for k, v in train_metrics.items()})
            self.logger.log_norms(t, "params", params)
            self.logger.log_norms(t, "grads", grads)
            if tangent is not None:
                self.logger.log_norms(t, "tangent", tangent)
            if log_eigs:
                sizes = [p.numel() for p in params.values()]
                for sign, k in ((1, self.n_top_eigs), (-1, self.n_neg_eigs)):
                    if k == 0:
                        continue
                    eigvals, eigvecs[sign], n_iter = top_hessian_eigs(
                        grads, params, k, eigvecs.get(sign),
                        precond=state.precond(t), seed=start, negative=sign < 0,
                    )
                    self.logger.log(
                        t, **{f"eig/{sign * (i + 1)}": lam for i, lam in enumerate(eigvals.unbind())},
                        **{f"lobpcg_iters/{sign}": torch.tensor(n_iter)},
                    )
                    for i, vec in enumerate(eigvecs[sign].unbind(dim=1)):
                        vec_dict = {
                            name: v.view_as(p) for v, (name, p) in zip(vec.split(sizes), params.items())
                        }
                        self.logger.log_norms(t, f"eigval/{sign * (i + 1)}", vec_dict)
            if tangent is not None and eigvecs:
                flat_tangent = torch.cat([t.flatten() for t in tangent.params.values()])
                if 1 in eigvecs:
                    dot_prod = torch.dot(flat_tangent, eigvecs[1][:, 0]) / flat_tangent.norm()
                    self.logger.log(t, tangent_top_eig=dot_prod)
                if -1 in eigvecs:
                    dot_prod = torch.dot(flat_tangent, eigvecs[-1][:, 0]) / flat_tangent.norm()
                    self.logger.log(t, tangent_neg_eig=dot_prod)
            if self.test_loader is not None and (
                (t - start + 1) % self.eval_every == 0 or t - start == n_steps - 1
            ):
                loader = self.test_loader
                if self.n_test is not None and t - start < n_steps - 1:  # random subset
                    loader = replace(
                        loader, indices=Subset(n=self.n_test, index=t).indices(loader.dataset)
                    )
                val_loss, val_metrics, _ = eval_loss(
                    self.model, self.loss_fn, state.params, loader, bf16=self.bf16,
                )
                self.logger.log(
                    t, test_loss=val_loss, **{f"test_{k}": v for k, v in val_metrics.items()}
                )
            del grads

        return state.apply(lambda t: t.cpu())

    def replay(
        self, init_state: OptimState, *, n_steps: int, start: int = 0, passes: int = 2, pbar: bool = True
    ) -> Iterator[OptimState]:
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
        eval_loaders: dict[str, EvalLoader], passes: int = 2,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """MAGIC: backpropagate the eval functions f = "mean loss over each eval loader"
        through the training steps [start, start + n_steps) from `init_state`.

        Returns: ({name: f(s_T)}, {name: df(s_T)/dw}).
        Cost: `passes + 1` trainings + one backward through training.
        Memory: `passes * n_steps^(1/passes) + #fns` states.
        """
        if (n_fns := len(eval_loaders)) == 0:
            return {}, {}
        states = self.replay(init_state, n_steps=n_steps, start=start, passes=passes)
        final_state = next(states).apply(lambda t: t.to(self.device))  # 1x training

        outputs: dict[str, Tensor] = {}  # f(s_T)
        betas: dict[str, Tensor] = {}    # df/dw
        delta = final_state.apply(lambda t: t.new_zeros(n_fns, *t.shape))  # df/ds_t
        for i, (name, eval_loader) in enumerate(eval_loaders.items()):
            f_value, _, f_grad = eval_loss(
                self.model, self.loss_fn, final_state.params,
                eval_loader, grad=True, bf16=self.bf16,
            )
            outputs[name] = f_value
            betas[name] = torch.zeros(len(self.loader.dataset), device=self.device)
            for d, g in dict_zip(delta.params, f_grad).values():
                d[i] = g

        steps = reversed(range(start, start + n_steps))
        for t, state in zip(steps, tqdm(states, total=n_steps, desc="magic")):
            X, Y, indices = self.loader.get(t)
            state = state.apply(lambda t: t.to(self.device, non_blocking=True).requires_grad_())
            weights = torch.ones_like(indices, dtype=X.dtype).requires_grad_()

            out = model_call(self.model, state.params, X, bf16=self.bf16)
            loss = (weights * self.loss_fn(out, Y)).mean()
            grads = torch.autograd.grad(loss, state.params, create_graph=True)
            new_state = state.step(grads, t)

            *d_state, d_weights = torch.autograd.grad(
                list(new_state), [*state, weights], list(delta), is_grads_batched=True
            )
            delta = delta.unflatten(d_state)
            for i, name in enumerate(eval_loaders):
                betas[name].index_add_(0, indices, d_weights[i])
                if self.logger is not None:
                    self.logger.log_norms(t, f"delta/{name}", delta.apply(lambda t: t[i]))

        return outputs, betas
