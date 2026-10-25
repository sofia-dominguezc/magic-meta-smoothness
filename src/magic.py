import math
import functools
from tqdm import tqdm
from dataclasses import dataclass, replace
from typing import Any, Callable, Concatenate, Generator, Iterator, Self, Sequence

import torch
from torch import Tensor, nn

from src.data import Loader
from src.logging import Logger
from src.optimizers import OptimState
from src.utils import HVP, LossFn, hvp_from_grads, metrics, model_call

Forward = Callable[[dict[str, Tensor], Tensor, Tensor], tuple[Tensor, Tensor]]  # (params, X, Y) -> (per-example losses, outputs)


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
    (a batch), `eval_loss` (many examples), `hvp_fn` (the batch Hessian).
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

    def _setup_forward(self, state: OptimState, weights: Tensor | None) -> tuple[Forward, OptimState, bool]:
        """`forward` vmapped if the state or the weights are batched (the state expanded to the weights if only they
        are), and compiled if `compile`: (forward_fn, state, batched)"""
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
        log: Logger | None = None, create_graph: bool = False, pbar: bool = True,
    ) -> Generator[tuple[int, OptimState, dict[str, Tensor] | None], None, OptimState]:
        """Train on steps [start, start + n_steps) and return the trained state.
        Batch mode:
            `weights` and/or `init_state` can be batched, logging is disabled.
        Logging mode:
            Log loss, train metrics, and norms of params/grads. No batching supported
        Yields (t, state_t, grads_t) for each step, and (T, state_T, None) at the end
        """
        state = init_state.apply(lambda t: t.to(self.device, non_blocking=True, copy=True))
        if weights is not None:
            weights = weights.to(self.device, non_blocking=True)
        forward_fn, state, batched = self._setup_forward(state, weights)
        create_graph = create_graph or state.needs_graph
        assert log is None or not batched, "can't log a batched run"
        assert not (batched and state.needs_graph), "can't vmap a second-order step"

        for t in tqdm(range(start, start + n_steps), disable=not pbar, desc="train"):
            X, Y, idx = self.loader.get_batch(t)
            params = {k: p.requires_grad_() for k, p in state.params.items()}
            losses, out = forward_fn(params, X, Y)  # [(R,) B], [(R,) B, C]
            loss = (losses if weights is None else weights[..., idx] * losses).mean(-1)  # [(R,)]
            grads = torch.autograd.grad(loss.sum(), params, create_graph=create_graph)

            if log is not None:
                train_metrics = {k: v.mean() for k, v in metrics(out.detach(), Y).items()}
                log.log(t, loss=loss.detach(), **train_metrics)
                log.log_norms(t, "params", params)
                log.log_norms(t, "grads", grads)
                if X.is_cuda:
                    log.log(t, gpu_gb=torch.tensor(torch.cuda.max_memory_allocated() / 2**30))
                    torch.cuda.reset_peak_memory_stats()

            if not create_graph:
                state = state.apply(lambda t: t.detach())
            yield t, state, grads
            state = state.step(t, grads).apply(lambda t: t.detach())

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

    @traceable
    def backprop(
        self, init_state: OptimState, *, n_steps: int, start: int = 0, eval_sets: Sequence[tuple[Tensor, Tensor]],
        passes: int = 2, log: Logger | None = None,
    ) -> Generator[tuple[int, OptimState, OptimState, Tensor], None, tuple[Tensor, Tensor]]:
        """MAGIC: backpropagate f_i = mean loss on the examples (X_i, Y_i) of `eval_sets` through steps
        [start, start + n_steps), all at once (batched cotangents: chunk the eval sets to bound the memory).
        Returns: (f(s_T) [#fns], df/dw [#fns, N])
        Yields (t, s_t, df/ds_t, df/dw_t [#fns, B])
        """
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

        steps = reversed(range(start, start + n_steps))
        for t, state in zip(steps, tqdm(states, total=n_steps, desc="magic")):
            X, Y, indices = self.loader.get_batch(t)
            state = state.apply(lambda t: t.to(self.device, non_blocking=True).requires_grad_())
            weights = torch.ones_like(indices, dtype=X.dtype).requires_grad_()

            loss = (weights * self.forward(state.params, X, Y)[0]).mean()
            grads = torch.autograd.grad(loss, state.params, create_graph=True)
            *d_state, d_weights = state.vjp(t, grads, [*state, weights], delta)

            delta = delta.unflatten(d_state)
            betas.index_add_(1, indices, d_weights)
            if log is not None:
                for i in range(n_fns):
                    log.log_norms(t, f"delta/{i}", delta.apply(lambda d: d[i]))
                    log.log(t, **{f"d_weights/{i}": d_weights[i].norm()})
            yield t, state, delta, d_weights

        return torch.stack(outputs), betas
