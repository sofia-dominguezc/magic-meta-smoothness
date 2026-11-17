import math
import functools
from functools import partial
from tqdm import tqdm
from dataclasses import dataclass, replace
from typing import Any, Callable, Concatenate, Generator, Iterator, Self, Sequence, overload

import torch
from torch import Tensor, nn

from src.data import Loader
from src.logging import Logger
from src.optimizers import OptimState, SecondOrderState, SGDState
from src.utils import HVP, OUTPUT_HESSIANS, LossFn, hvp_from_grads, metrics, model_call


class traceable[**P, Y, R]:
    """Decorator of a generator method `g`, which then also runs as a plain function:
    - `obj.g(...)`: runs it to the end, returns its return value
    - `obj.g.trace(...)`: the generator, i.e. the yielded values
    The purpose is that extra logging can be attached for free to the given function.
    """

    def __init__(self, g: Callable[Concatenate[Any, P], Generator[Y, None, R]], obj: Any = None):
        self.g, self.obj = g, obj
        functools.update_wrapper(self, g)

    def __get__(self, obj: Any, objtype: type | None = None) -> Self:
        return self if obj is None else type(self)(self.g, obj)

    def trace(self, *args: P.args, **kwargs: P.kwargs) -> Iterator[Y]:
        return self.g(self.obj, *args, **kwargs)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        gen = self.g(self.obj, *args, **kwargs)
        while True:
            try:
                next(gen)
            except StopIteration as e:
                return e.value


@dataclass(frozen=True, kw_only=True)
class Magic:
    """Training of `model` on `loader`, REPLAY and MAGIC, by stateful autograd. All the forward passes: `forward`
    (a batch), `eval_loss` (many examples), `hvp_fn` / `gnvp_fn` (the batch Hessian / Gauss-Newton), `batched_grads_fn`
    (per-example gradients).
    - `bf16`: the model forward in bfloat16 (autocast)
    - `compile`: torch.compile the training forward
    - `train` and `backprop` are `traceable`: `.trace(...)` for their intermediate values
    """

    model: nn.Module
    loader: Loader
    loss_fn: LossFn
    bf16: bool = False
    compile: bool = False

    @property
    def device(self) -> torch.device | str | None:
        return self.loader.device

    def forward(
        self, params: dict[str, Tensor], X: Tensor, Y: Tensor, weights: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """The model on a batch: (mean loss, with the example `weights` [B] if given; outputs [B, C])"""
        out = model_call(self.model, params, X, bf16=self.bf16)
        losses = self.loss_fn(out, Y)
        return (losses if weights is None else weights * losses).mean(), out

    def _train_forward(self, state: OptimState, weights: Tensor | None) -> tuple[Callable, OptimState, bool]:
        """Compile and vmap Magic.forward if specified. `weights`: [N] loss weights of the training examples
        Batch mode (first-order optimizers): `weights` [R, N] and/or a batched `init_state` train R models, vmapped
        """
        name, p = next(iter(state.params.items()))
        batched = p.ndim > self.model.get_parameter(name).ndim
        if weights is not None and weights.ndim == 2 and not batched:
            state, batched = state.apply(lambda t: t.expand(len(weights), *t.shape).clone()), True
        assert not (batched and isinstance(state, SecondOrderState)), "can't vmap a second-order step"
        forward_fn = self.forward
        if batched:
            vmapped = torch.func.vmap(forward_fn, in_dims=(0, None, None, None if weights is None else 0))
            forward_fn = lambda params, X, Y, weights=None: vmapped(params, X, Y, weights)
        if self.compile:
            forward_fn = torch.compile(forward_fn)
        return forward_fn, state, batched

    @torch.no_grad()
    def eval_loss(self, params: dict[str, Tensor], loader: Loader, indices: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Mean loss and metrics of `params` on the examples `indices` of `loader`, accumulated batch by batch"""
        n = len(indices)
        loss, mean_metrics = torch.zeros((), device=loader.device), {}
        for X, Y in loader.iter_batches(indices):
            batch_loss, out = self.forward(params, X, Y)
            loss += batch_loss * len(X) / n
            for k, v in metrics(out, Y).items():
                mean_metrics[k] = mean_metrics.get(k, 0) + v.sum() / n
        return loss, mean_metrics

    def hvp_fn(self, params: dict[str, Tensor], X: Tensor, Y: Tensor) -> HVP:
        """`hvp_from_grads` of the mean loss of the batch (X, Y) at `params`"""
        params = {k: p.detach().requires_grad_() for k, p in params.items()}
        grads = torch.autograd.grad(self.forward(params, X, Y)[0], params, create_graph=True)
        return hvp_from_grads(grads, params)

    @traceable
    def train(
        self, init_state: OptimState, *, n_steps: int, start: int = 0, weights: Tensor | None = None,
        log: Logger | None = None, create_graph: bool = False, hook: Callable[[OptimState], OptimState] | None = None,
        pbar: bool = True,
    ) -> Generator[tuple[int, OptimState, dict[str, Tensor] | None], None, OptimState]:
        """Train on steps [start, start + n_steps) and return the trained state.
        `create_graph`: yield the grads with their graph (for HVPs)
        `hook`: replaces the detaching of the state after each step (and is applied to the initial state)
        Yields (t, state_t, grads_t) for each step, and (T, state_T, None) at the end
        """
        state = init_state.apply(lambda t: t.to(self.device, non_blocking=True, copy=True))
        if weights is not None:
            weights = weights.to(self.device, non_blocking=True)
        forward_fn, state, batched = self._train_forward(state, weights)
        second_order = isinstance(state, SecondOrderState)  # (its grads always have their graph)
        keep_graph = create_graph or second_order or hook is not None
        if hook is not None:
            state = hook(state)

        for t in tqdm(range(start, start + n_steps), disable=not pbar, desc="train"):
            X, Y, idx = self.loader.get_batch(t)
            params = {k: p.requires_grad_() for k, p in state.params.items()}
            forward_step = partial(forward_fn, X=X, Y=Y, weights=None if weights is None else weights[..., idx])

            next_state, grads, loss, out = state.step(t, forward_step, create_graph=create_graph and not second_order)
            yield t, state if keep_graph else state.apply(lambda t: t.detach()), grads
            state = next_state.apply(lambda t: t.detach()) if hook is None else hook(next_state)

            if log is not None:
                metrics_fn = torch.func.vmap(metrics, in_dims=(0, None)) if batched else metrics
                train_metrics = {k: v.mean(-1) for k, v in metrics_fn(out.detach(), Y).items()}
                log.log(t, loss=loss.detach(), **train_metrics)
                if not batched:
                    log.log_norms(t, "params", params)
                    log.log_norms(t, "grads", grads)
                if isinstance(state, SecondOrderState):
                    log.log(t, **state.stats)
                if X.is_cuda:
                    log.log(t, gpu_gb=torch.tensor(torch.cuda.max_memory_allocated() / 2**30))
                    torch.cuda.reset_peak_memory_stats()

        yield start + n_steps, state, None
        return state.apply(lambda t: t.detach().cpu())

    def replay(
        self, init_state: OptimState, *, n_steps: int, start: int = 0, passes: int = 2, pbar: bool = True
    ) -> Iterator[OptimState]:
        """Reconstruct the states [start, start + n_steps] in reverse order
        Cost: `passes` trainings. Memory: `passes * n_steps^(1/passes)` states.
        """
        if n_steps == 0:
            yield init_state
            return
        chunk_size = math.ceil(n_steps ** ((passes - 1) / passes))
        checkpoints = [
            state.apply(lambda t: t.to("cpu", copy=True))
            for t, state, _ in self.train.trace(init_state, n_steps=n_steps, start=start, pbar=pbar)
            if (t - start) % chunk_size == 0 or t == start + n_steps
        ]
        yield checkpoints.pop()  # final state
        for i, ckpt in reversed(list(enumerate(checkpoints))):
            r_start = start + i * chunk_size
            r_steps = min(chunk_size, start + n_steps - r_start)
            yield from self.replay(
                ckpt, n_steps=r_steps - 1, start=r_start, passes=passes - 1, pbar=False
            )

    def gnvp_fn(self, params: dict[str, Tensor], X: Tensor, Y: Tensor) -> HVP:
        """Like `hvp_fn` for the Gauss-Newton matrix G = J^T H_L J / B.
        No second derivatives, so it vmaps over the vectors and can be compiled"""
        params = {k: p.detach() for k, p in params.items()}
        model_fn = lambda p: model_call(self.model, p, X, bf16=self.bf16)
        output_hessian = OUTPUT_HESSIANS[self.loss_fn]

        def gnvp(V: dict[str, Tensor], create_graph: bool = False) -> dict[str, Tensor]:
            out, vjp = torch.func.vjp(model_fn, params)[:2]

            def gnv(v: dict[str, Tensor]) -> dict[str, Tensor]:
                Jv = torch.func.jvp(model_fn, (params,), (v,))[1]  # [B, C]
                return vjp(output_hessian(out, Y, Jv) / len(X))[0]

            batched = next(iter(V.values())).ndim > next(iter(params.values())).ndim
            return torch.func.vmap(gnv)(V) if batched else gnv(V)

        return torch.compile(gnvp) if self.compile else gnvp

    def batched_grads_fn(self, params: dict[str, Tensor], X: Tensor, Y: Tensor) -> dict[str, Tensor]:
        """Per-example gradients {name: [B, *shape]} of the batch (X, Y) at `params`"""
        def example_loss(params: dict[str, Tensor], x: Tensor, y: Tensor) -> Tensor:
            return self.forward(params, x[None], y[None])[0]

        batched_grads = torch.func.vmap(torch.func.grad(example_loss), in_dims=(None, 0, 0))
        return (torch.compile(batched_grads) if self.compile else batched_grads)(params, X, Y)

    @traceable
    def backprop(
        self, init_state: OptimState, *, n_steps: int, start: int = 0, eval_sets: Sequence[tuple[Tensor, Tensor]],
        passes: int = 2, gn: bool = False, perturbed: range | None = None, log: Logger | None = None,
    ) -> Generator[tuple[int, OptimState, OptimState, Tensor], None, tuple[Tensor, Tensor]]:
        """MAGIC: backpropagate f_i = mean loss on the examples (X_i, Y_i) of `eval_sets` through steps
        [start, start + n_steps), all at once (batched cotangents: chunk the eval sets to bound the memory).
        Returns: (f(s_T) [#fns], df/dw [#fns, N])
        Yields (t, s_t, df/ds_t, df/dw_t [#fns, B]) for the steps t >= min(perturbed)
        """
        perturbed = range(start, start + n_steps) if perturbed is None else perturbed
        assert perturbed.step == 1 and start <= perturbed.start <= perturbed.stop <= start + n_steps, (perturbed, start, n_steps)

        n_fns = len(eval_sets)
        replayed_states = self.replay(init_state, n_steps=n_steps, start=start, passes=passes)
        final_state = next(replayed_states).apply(lambda t: t.to(self.device).requires_grad_())

        outputs: list[Tensor] = []  # f(s_T)
        f_grads: list[dict[str, Tensor]] = []  # df/ds_T
        for X, Y in eval_sets:
            f = self.forward(final_state.params, X.to(self.device), Y.to(self.device))[0]
            outputs.append(f.detach())
            f_grads.append(torch.autograd.grad(f, final_state.params))
        delta = replace(  # df/ds_t [#fns, ...]
            final_state.apply(lambda t: t.new_zeros(n_fns, *t.shape)),
            params={k: torch.stack([g[k] for g in f_grads]) for k in final_state.params}
        )
        betas = torch.zeros(n_fns, len(self.loader.dataset), device=self.device)  # df/dw

        pbar = tqdm(total=start + n_steps - perturbed.start, desc="magic")
        for t, state in zip(reversed(range(start, start + n_steps)), replayed_states):
            if t < perturbed.start:
                break
            pbar.update()
            X, Y, indices = self.loader.get_batch(t)
            state = state.apply(lambda t: t.to(self.device, non_blocking=True).requires_grad_())

            if gn:
                assert type(state) is SGDState, "the GN backward is only implemented for SGD"
                batched_grads = self.batched_grads_fn(state.params, X, Y)  # detached
                delta, d_grads = state.gn_vjp(t, delta, self.gnvp_fn(state.params, X, Y))
                d_weights = sum(d_grads[k].flatten(1) @ g.flatten(1).T for k, g in batched_grads.items()) / len(X)  # [#fns, B]
            else:
                weights = torch.ones_like(indices, dtype=X.dtype).requires_grad_()
                forward_fn = partial(self.forward, X=X, Y=Y, weights=weights)
                *d_state, d_weights = state.vjp(t, forward_fn, [*state, weights], delta)
                delta = delta.unflatten(d_state)
            if t in perturbed:
                betas.index_add_(1, indices, d_weights)

            if log is not None:
                params = {k: p.detach() for k, p in state.params.items()}
                for i, (X_i, Y_i) in enumerate(eval_sets):
                    with torch.no_grad():
                        f_t = self.forward(params, X_i.to(self.device), Y_i.to(self.device))[0]
                    log.log(t, **{f"f/{i}": f_t})
                    log.log_norms(t, f"delta/{i}", delta.apply(lambda d: d[i]))
                    log.log(t, **{f"d_weights/{i}": d_weights[i].norm()})
            yield t, state, delta, d_weights
        pbar.close()

        return torch.stack(outputs), betas
