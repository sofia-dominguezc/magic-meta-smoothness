import math
from tqdm import tqdm
from typing import Callable, Iterator
from dataclasses import dataclass, replace
from functools import partial

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD
from torch.func import grad_and_value, functional_call

from src.data import EvalLoader, Loader
from src.logging import Logger, op_norm
from src.eigvals import top_hessian_eigs
from src.models import OptimState, dict_zip


LossFn = Callable[[Tensor, Tensor], Tensor]  # (model output [B, C], int labels [B]) -> per-example losses [B]


def cross_entropy(out: Tensor, Y: Tensor) -> Tensor:
    return F.cross_entropy(out, Y, reduction="none")


def mse(out: Tensor, Y: Tensor) -> Tensor:
    """Mean over the classes of the squared error to the one-hot labels."""
    return (out - F.one_hot(Y, out.shape[1]).to(out.dtype)).pow(2).mean(1)


LOSSES: dict[str, LossFn] = {"ce": cross_entropy, "mse": mse}


def metrics(out: Tensor, Y: Tensor) -> dict[str, Tensor]:
    """Per-example classification metrics of the model output, for logging: accuracy and margin."""
    correct = out.gather(1, Y[:, None])[:, 0]
    best_other = out.scatter(1, Y[:, None], float("-inf")).max(1).values
    return {"acc": (out.argmax(1) == Y).to(out.dtype), "margin": correct - best_other}


def model_call(
    model: nn.Module, params: dict[str, Tensor], X: Tensor, *, dtype: torch.dtype | None = None
) -> Tensor:
    """`model(params, X)` computed in `dtype` (bfloat16 also uses autocast: e.g. softmax stays in float32)."""
    if dtype is None:
        return functional_call(model, params, (X,))
    orig_dtype = next(iter(params.values())).dtype
    with torch.autocast(X.device.type, torch.bfloat16, enabled=dtype == torch.bfloat16):
        out: Tensor = functional_call(
            model, {k: p.to(dtype) for k, p in params.items()}, (X.to(dtype),)
        )
    return out.to(orig_dtype)


def eval_loss(
    model: nn.Module, loss_fn: LossFn, params: dict[str, Tensor], loader: EvalLoader, grad: bool = False,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    """Mean loss over the examples of `loader`, accumulated batch by batch to bound the memory (`model_call` in `dtype`).
    Returns: (loss, {metric: mean value}, {param: d loss / d param} if `grad` else {})."""

    def batch_loss(params: dict[str, Tensor], X: Tensor, Y: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """This batch's share of the mean loss, and of the mean metrics."""
        out = model_call(model, params, X, dtype=dtype)
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
    """

    model: nn.Module
    loader: Loader
    loss_fn: LossFn
    logger: Logger | None = None
    test_loader: EvalLoader | None = None
    eval_every: int = 10
    n_eigs: int = 1
    log_tangent: bool = True
    compute_dtype: torch.dtype | None = None
    compile: bool = False
    chunk_size: int | None = None

    @property
    def device(self) -> torch.device | str | None:
        return self.loader.device

    @partial(torch.func.grad, argnums=1)
    def _grad(self, params: dict[str, Tensor], weights: Tensor, X: Tensor, Y: Tensor) -> Tensor:
        """The training objective of one step with samples weighted by `weights`"""
        out = model_call(self.model, params, X, dtype=self.compute_dtype)
        return (weights * self.loss_fn(out, Y)).mean()

    def train(
        self, init_state: OptimState, *, n_steps: int, start: int = 0,
        weights: Tensor | None = None, weights_from: int = 0, ckpt_every: int | None = None,
    ) -> tuple[OptimState, list[OptimState]]:
        """Train on steps [start, start + n_steps) with the per-example weights `weights`.
        `weights` can be None (all ones) or (#n_weights, #datapoints).
        Returns: (final state, states (start, start + ckpt_every, ...))
        """
        state = init_state.apply(lambda t: t.to(self.device, non_blocking=True, copy=True))
        if weights is None:
            weights = torch.ones(len(self.loader.dataset)).to(self.device, non_blocking=True)
            grad_fn = self._grad
        else:
            weights = weights.to(self.device, non_blocking=True)
            state = state.apply(lambda t: t.expand(len(weights), *t.shape).clone())
            grad_fn = torch.func.vmap(
                self._grad, in_dims=(0, 0, None, None), chunk_size=self.chunk_size
            )
        if self.compile:
            grad_fn = torch.compile(grad_fn)
        ckpts: list[OptimState] = []
        for t in range(start, start + n_steps):
            if ckpt_every and (t - start) % ckpt_every == 0:
                ckpts.append(state.apply(lambda t: t.to("cpu", copy=True)))
            X, Y, idx = self.loader.get(t)
            w = weights[..., idx] if t >= weights_from else torch.ones_like(weights[..., idx])
            grads = grad_fn(state.params, w, X, Y)
            state = state.step(grads)
        return state.apply(lambda t: t.cpu()), ckpts

    def train_logged(self, init_state: OptimState, *, n_steps: int, start: int = 0) -> OptimState:
        """Train on all the data on steps [start, start + n_steps) while logging.
        Logs the top few eigenvalues, the norm of ds_t/ds_0 on a random direction, and more.
        """
        assert self.logger is not None
        state = init_state.apply(lambda t: t.to(self.device, copy=True))

        eigvecs: Tensor | None = None
        tangent: OptimState | None = None  # ds_t/ds_0 @ v
        if self.log_tangent:
            g = torch.Generator().manual_seed(start)
            tangent = state.apply(lambda t: torch.randn(t.shape, generator=g, dtype=t.dtype))
            tangent = tangent.apply(lambda t: (t / op_norm(t, n_iter=10)[0]).to(self.device))

        for t in tqdm(range(start, start + n_steps)):
            X, Y, _ = self.loader.get(t)
            if tangent is not None:
                fwAD.enter_dual_level()
                state = state.unflatten([fwAD.make_dual(s, d) for s, d in zip(state, tangent)])
            params = {name: p.requires_grad_() for name, p in state.params.items()}

            out = model_call(self.model, params, X, dtype=self.compute_dtype)
            loss = self.loss_fn(out, Y).mean()
            grads = torch.autograd.grad(loss, params, create_graph=self.n_eigs > 0)
            state = state.step(grads)

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
            if self.n_eigs > 0 and t % 10 == 0:
                eigvals, eigvecs, n_iter = top_hessian_eigs(
                    grads, params, self.n_eigs, eigvecs, precond=state.precond, seed=start
                )
                eigvals = eigvals * (2 / state.stability_threshold)
                self.logger.log(
                    t, **{f"eig/{i + 1}": v for i, v in enumerate(eigvals.unbind())},
                    lobpcg_iters=torch.tensor(n_iter),
                )
            if tangent is not None and eigvecs is not None:
                flat_tangent = torch.cat([t.flatten() for t in tangent.params.values()])
                dot_prod = torch.dot(flat_tangent, eigvecs[:, 0]) / flat_tangent.norm()
                self.logger.log(t, tangent_top_eig=dot_prod)
            if self.test_loader is not None and (
                (t - start + 1) % self.eval_every == 0 or t - start == n_steps - 1
            ):
                val_loss, val_metrics, _ = eval_loss(
                    self.model, self.loss_fn, state.params,
                    self.test_loader, dtype=self.compute_dtype,
                )
                self.logger.log(
                    t, test_loss=val_loss, **{f"test_{k}": v for k, v in val_metrics.items()}
                )
            del grads

        return state.apply(lambda t: t.cpu())

    def rewind(
        self, init_state: OptimState, *, n_steps: int, start: int = 0, passes: int = 2
    ) -> Iterator[OptimState]:
        """Reconstruct the states [start, start + n_steps] in reverse order.
        Cost: `passes` trainings. Memory: `passes * n_steps^(1/passes)` states.
        """
        chunk = math.ceil(n_steps ** ((passes - 1) / passes))  # 1 if passes == 1
        final, ckpts = self.train(init_state, n_steps=n_steps, start=start, ckpt_every=chunk)
        yield final
        if passes == 1:
            yield from reversed(ckpts)
            return
        for i, ckpt in reversed(list(enumerate(ckpts))):
            t0 = start + i * chunk
            inner = self.rewind(
                ckpt, n_steps=min(chunk, start + n_steps - t0), start=t0, passes=passes - 1
            )
            next(inner)  # its final state, already yielded
            yield from inner

    def backprop(
        self, init_state: OptimState, *, n_steps: int, start: int = 0,
        eval_loaders: dict[str, EvalLoader], passes: int = 2,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """MAGIC: backpropagate the eval functions f = "mean loss over each eval loader"
        through the training steps [start, start + n_steps) from `init_state`.

        Returns: ({name: f(s_T)}, {name: df(s_T)/dw}).
        Cost: `1 + passes + #fns` trainings; memory: `passes * n_steps^(1/passes) + #fns` states.
        """
        if (n_fns := len(eval_loaders)) == 0:
            return {}, {}
        states = self.rewind(init_state, n_steps=n_steps, start=start, passes=passes)
        final_state = next(states).apply(lambda t: t.to(self.device))  # 1x training

        def backprop_step(
            state: OptimState, X: Tensor, Y: Tensor, delta: dict[str, dict[str, Tensor]]
        ) -> tuple[dict[str, dict[str, Tensor]], Tensor]:
            """(df/ds_t, df/dw_t) = ds_{t+1} / d(s_t, w_t) @ df/ds_{t+1} for all `f`.
            Cost: `1 + #fns` trainings
            """

            def step_fn(
                param_groups: dict[str, dict[str, Tensor]], weights: Tensor
            ) -> dict[str, dict[str, Tensor]]:
                s = replace(state, **param_groups)
                return s.step(self._grad(s.params, weights, X, Y)).param_groups

            weights = torch.ones(len(Y), dtype=X.dtype, device=X.device)
            _, vjp_fn = torch.func.vjp(step_fn, state.param_groups, weights)
            return torch.func.vmap(vjp_fn, chunk_size=self.chunk_size)(delta)

        if self.compile:  # NOTE: it may break on Adam because of `t`
            backprop_step = torch.compile(backprop_step)

        outputs: dict[str, Tensor] = {}  # f(s_T)
        betas: dict[str, Tensor] = {}    # df/dw
        delta = final_state.apply(lambda t: t.new_zeros(n_fns, *t.shape))  # df/ds_t
        for i, (name, eval_loader) in enumerate(eval_loaders.items()):
            f_value, _, f_grad = eval_loss(
                self.model, self.loss_fn, final_state.params,
                eval_loader, grad=True, dtype=self.compute_dtype,
            )
            outputs[name] = f_value
            betas[name] = torch.zeros(len(self.loader.dataset), device=self.device)
            for t, g in dict_zip(delta.params, f_grad).values():
                t[i] = g

        steps = reversed(range(start, start + n_steps))
        for t, state in zip(steps, tqdm(states, total=n_steps)):  # `passes - 1` trainings
            X, Y, indices = self.loader.get(t)
            state = state.apply(lambda t: t.to(self.device, non_blocking=True))
            d_state, d_weights = backprop_step(state, X, Y, delta.param_groups)
            delta = replace(delta, **d_state)
            for i, name in enumerate(eval_loaders):
                betas[name].index_add_(0, indices, d_weights[i])
                if self.logger is not None:
                    self.logger.log_norms(t, f"delta/{name}", delta.apply(lambda t: t[i]))

        return outputs, betas
