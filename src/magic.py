import math
import functools
from tqdm import tqdm
from dataclasses import dataclass, replace
from typing import Any, Callable, Concatenate, Generator, Iterator, Self, Sequence, overload

import torch
from torch import Tensor, nn

from src.data import Loader
from src.logging import Logger
from src.optimizers import OptimState, SecondOrderState
from src.utils import HVP, OUTPUT_HESSIANS, LossFn, hvp_from_grads, metrics, model_call

Forward = Callable[[dict[str, Tensor], Tensor, Tensor], tuple[Tensor, Tensor]]  # (params, X, Y) -> (per-example losses, outputs)
Backward = Callable[[dict[str, Tensor], Tensor, Tensor], dict[str, Tensor]]


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

    def forward(self, params: dict[str, Tensor], X: Tensor, Y: Tensor) -> tuple[Tensor, Tensor]:
        """The model on a batch: (per-example losses [B], outputs [B, C])"""
        out = model_call(self.model, params, X, bf16=self.bf16)
        return self.loss_fn(out, Y), out

    @torch.no_grad()
    def eval_loss(self, params: dict[str, Tensor], loader: Loader, indices: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Mean loss and metrics of `params` on the examples `indices` of `loader`, accumulated batch by batch"""
        n = len(indices)
        loss, mean_metrics = torch.zeros((), device=loader.device), {}
        for X, Y in loader.iter_batches(indices):
            losses, out = self.forward(params, X, Y)
            loss += losses.sum() / n
            for k, v in metrics(out, Y).items():
                mean_metrics[k] = mean_metrics.get(k, 0) + v.sum() / n
        return loss, mean_metrics

    def hvp_fn(self, params: dict[str, Tensor], X: Tensor, Y: Tensor) -> HVP:
        """`hvp_from_grads` of the mean loss of the batch (X, Y) at `params`"""
        params = {k: p.detach().requires_grad_() for k, p in params.items()}
        grads = torch.autograd.grad(self.forward(params, X, Y)[0].mean(), params, create_graph=True)
        return hvp_from_grads(grads, params)

    def forward_fn(self, state: OptimState, weights: Tensor | None) -> tuple[Forward, OptimState, bool]:
        """`forward` vmapped if the state or the weights are batched (the state expanded to
        the weights if only they are), and compiled if `compile`"""
        forward_fn = self.forward
        name, p = next(iter(state.params.items()))
        batched_state = p.ndim > self.model.get_parameter(name).ndim
        batched_weights = weights is not None and weights.ndim == 2
        if batched_weights and not batched_state:
            assert weights is not None
            state = state.apply(lambda t: t.expand(len(weights), *t.shape).clone())
        if batched_weights or batched_state:
            forward_fn = torch.func.vmap(forward_fn, in_dims=(0, None, None))
        if self.compile:
            forward_fn = torch.compile(forward_fn)
        return forward_fn, state, batched_weights or batched_state

    @traceable
    def train(
        self, init_state: OptimState, *, n_steps: int, start: int = 0, weights: Tensor | None = None,
        log: Logger | None = None, create_graph: bool = False, hook: Callable[[OptimState], OptimState] | None = None,
        pbar: bool = True,
    ) -> Generator[tuple[int, OptimState, dict[str, Tensor] | None], None, OptimState]:
        """Train on steps [start, start + n_steps) and return the trained state.
        Batch mode:
            `weights` and/or `init_state` can be batched; only the scalars are logged, as [R] vectors
        Logging mode:
            Log loss, train metrics, norms of params/grads and `stats`
        Yields (t, state_t, grads_t) for each step, and (T, state_T, None) at the end
        `hook`: replaces the detaching of the state after each step (and is applied to the initial state)
        """
        state = init_state.apply(lambda t: t.to(self.device, non_blocking=True, copy=True))
        if weights is not None:
            weights = weights.to(self.device, non_blocking=True)
        forward_fn, state, batched = self.forward_fn(state, weights)
        second_order = isinstance(state, SecondOrderState)  # (the step needs the graph of the gradients)
        create_graph = create_graph or second_order
        assert not (batched and second_order), "can't vmap a second-order step"
        if hook is not None:
            state = hook(state)

        for t in tqdm(range(start, start + n_steps), disable=not pbar, desc="train"):
            X, Y, idx = self.loader.get_batch(t)
            params = {k: p.requires_grad_() for k, p in state.params.items()}
            losses, out, aux_loss = state.forward(forward_fn, X, Y)  # [(R,) B], [(R,) B, C]

            loss = (losses if weights is None else weights[..., idx] * losses).mean(-1)  # [(R,)]
            if aux_loss is not None:
                aux_loss = (aux_loss if weights is None else weights[..., idx] * aux_loss).mean(-1)
            grads = torch.autograd.grad(loss.sum(), params, create_graph=create_graph)

            if not create_graph and hook is None:
                state = state.apply(lambda t: t.detach())
            yield t, state, grads
            state = state.step(t, grads, aux_loss)
            state = state.apply(lambda t: t.detach()) if hook is None else hook(state)

            if log is not None:
                metrics_fn = torch.func.vmap(metrics, in_dims=(0, None)) if batched else metrics
                train_metrics = {k: v.mean(-1) for k, v in metrics_fn(out.detach(), Y).items()}  # [(R,)]
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
            return self.forward(params, x[None], y[None])[0].squeeze(0)

        batched_grads = torch.func.vmap(torch.func.grad(example_loss), in_dims=(None, 0, 0))
        return (torch.compile(batched_grads) if self.compile else batched_grads)(params, X, Y)

    @traceable
    def backprop(
        self, init_state: OptimState, *, n_steps: int, start: int = 0, eval_sets: Sequence[tuple[Tensor, Tensor]],
        passes: int = 2, gn_below: int | None = None, skip_below: int | None = None, log: Logger | None = None,
    ) -> Generator[tuple[int, OptimState, OptimState, Tensor], None, tuple[Tensor, Tensor]]:
        """MAGIC: backpropagate f_i = mean loss on the examples (X_i, Y_i) of `eval_sets` through steps
        [start, start + n_steps), all at once (batched cotangents: chunk the eval sets to bound the memory).
        Returns: (f(s_T) [#fns], df/dw [#fns, N])
        Yields (t, s_t, df/ds_t, df/dw_t [#fns, B]) for the steps t >= `skip_below`
        Logs "f/<i>" = f_i(s_t), the norms of df/ds_t ("delta/<i>") and |df/dw_t| ("d_weights/<i>") at each step.
        """
        gn_below = start if gn_below is None else gn_below
        end = max(start, start if skip_below is None else skip_below)  # the first step of the backward's range
        assert end <= max(gn_below, end) <= start + n_steps, (skip_below, gn_below)

        n_fns = len(eval_sets)
        states = self.replay(init_state, n_steps=n_steps, start=start, passes=passes)
        final_state = next(states).apply(lambda t: t.to(self.device).requires_grad_())

        outputs: list[Tensor] = []  # f(s_T)
        f_grads: list[dict[str, Tensor]] = []  # df/ds_T
        for X, Y in eval_sets:
            f = self.forward(final_state.params, X.to(self.device), Y.to(self.device))[0].mean()
            outputs.append(f.detach())
            f_grads.append(torch.autograd.grad(f, final_state.params))
        delta = replace(  # df/ds_t
            final_state.apply(lambda t: t.new_zeros(n_fns, *t.shape)),
            params={k: torch.stack([g[k] for g in f_grads]) for k in final_state.params}
        )
        betas = torch.zeros(n_fns, len(self.loader.dataset), device=self.device)  # df/dw
        if log is not None:
            for i in range(n_fns):
                log.log_norms(start + n_steps, f"delta/{i}", delta.apply(lambda d: d[i]))
                log.log(start + n_steps, **{f"f/{i}": outputs[i]})

        pbar = tqdm(total=start + n_steps - end, desc="magic")
        for t, state in zip(reversed(range(start, start + n_steps)), states):
            if t < end:
                break
            pbar.update()
            X, Y, indices = self.loader.get_batch(t)
            state = state.apply(lambda t: t.to(self.device, non_blocking=True).requires_grad_())
            weights = torch.ones_like(indices, dtype=X.dtype).requires_grad_()

            if t < gn_below:
                assert not isinstance(state, SecondOrderState)
                assert type(state).forward is OptimState.forward, "the GN backward takes the gradients at the params"
                batched_grads = self.batched_grads_fn(state.params, X, Y)  # detached
                grads = {k: torch.tensordot(weights, g, dims=1) / len(X) for k, g in batched_grads.items()}
                delta, (d_weights,) = state.gn_vjp(t, grads, [weights], delta, self.gnvp_fn(state.params, X, Y))
            else:
                losses, _, aux_loss = state.forward(self.forward, X, Y)
                loss, aux_loss = (weights * losses).mean(), None if aux_loss is None else (weights * aux_loss).mean()
                grads = torch.autograd.grad(loss, state.params, create_graph=True)
                *d_state, d_weights = state.vjp(t, grads, [*state, weights], delta, aux_loss)
                delta = delta.unflatten(d_state)

            betas.index_add_(1, indices, d_weights)
            if log is not None:
                params = {k: p.detach() for k, p in state.params.items()}
                for i, (X_i, Y_i) in enumerate(eval_sets):
                    with torch.no_grad():
                        f_t = self.forward(params, X_i.to(self.device), Y_i.to(self.device))[0].mean()
                    log.log(t, **{f"f/{i}": f_t})
                    log.log_norms(t, f"delta/{i}", delta.apply(lambda d: d[i]))
                    log.log(t, **{f"d_weights/{i}": d_weights[i].norm()})
            yield t, state, delta, d_weights
        pbar.close()

        return torch.stack(outputs), betas
