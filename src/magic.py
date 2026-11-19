import math
import functools
from functools import partial
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Callable, Concatenate, Generator, Iterator, Self, overload

import torch
from torch import Tensor

from src.data import Loader
from src.logging import Logger
from src.objective import Objective, metrics
from src.optimizers import OptimState
from src.second_order import SecondOrderState


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
    """Training on the batches of `loader` with the loss `objective`, forward (`train`, `replay`) and backward (`backprop`: MAGIC).
    - `compile`: torch.compile the training forward
    - `train` and `backprop` are `traceable`: `.trace(...)` for their intermediate values
    """

    objective: Objective
    loader: Loader
    compile: bool = False

    @property
    def device(self) -> torch.device | str | None:
        return self.loader.device

    def _train_forward(self, state: OptimState, weights: Tensor | None) -> tuple[Callable, OptimState, bool]:
        """Compile and vmap the objective's forward if specified. `weights`: [N] loss weights of the training examples
        Batch mode (first-order optimizers): `weights` [R, N] and/or a batched `init_state` train R models, vmapped
        """
        name, p = next(iter(state.params.items()))
        batched = p.ndim > self.objective.model.get_parameter(name).ndim
        if weights is not None and weights.ndim == 2 and not batched:
            state, batched = state.apply(lambda t: t.expand(len(weights), *t.shape).clone()), True
        assert not (batched and isinstance(state, SecondOrderState)), "can't vmap a second-order step"
        forward_fn = self.objective.forward
        if batched:
            vmapped = torch.func.vmap(forward_fn, in_dims=(0, None, None, None if weights is None else 0))
            forward_fn = lambda params, X, Y, weights=None: vmapped(params, X, Y, weights)
        if self.compile and not isinstance(state, SecondOrderState):
            forward_fn = torch.compile(forward_fn)
        return forward_fn, state, batched

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
        second_order = isinstance(state, SecondOrderState)  # (its grads always have their graph)
        if weights is None and second_order:  # to avoid CentralFlow re-compilations
            weights = torch.ones(len(self.loader.dataset), device=self.device, dtype=next(iter(state.params.values())).dtype)
        forward_fn, state, batched = self._train_forward(state, weights)
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
            yield from self.replay(ckpt, n_steps=r_steps - 1, start=r_start, passes=passes - 1, pbar=False)

    @traceable
    def backprop(
        self, init_state: OptimState, d_state: OptimState, *, n_steps: int, start: int = 0, passes: int = 2
    ) -> Generator[tuple[int, OptimState, OptimState, Tensor], None, tuple[OptimState, Tensor]]:
        """MAGIC: backpropagate `d_state` = df/ds_{start + n_steps} (batched over the fns, see `OptimState.cotangent`)
        Returns: (df/ds_start, df/dw [#fns, N])
        Yields (t, s_t, df/ds_t, df/dw_t [#fns, B]) for each step
        """
        d_state = d_state.apply(lambda t: t.to(self.device))
        n_fns = len(next(iter(d_state)))
        replayed_states = self.replay(init_state, n_steps=n_steps, start=start, passes=passes)
        next_state = next(replayed_states).apply(lambda t: t.to(self.device))
        d_weights = torch.zeros(n_fns, len(self.loader.dataset), device=self.device)  # df/dw

        pbar = tqdm(total=n_steps, desc="magic")
        for t, state in zip(reversed(range(start, start + n_steps)), replayed_states):
            pbar.update()
            X, Y, indices = self.loader.get_batch(t)
            state = state.apply(lambda t: t.detach().to(self.device, non_blocking=True).requires_grad_())

            weights = torch.ones_like(indices, dtype=X.dtype).requires_grad_()
            forward_fn = partial(self.objective.forward, X=X, Y=Y, weights=weights)
            *d_inputs, d_batch_weights = state.vjp(t, forward_fn, [*state, weights], d_state, next_state)
            d_state = d_state.unflatten(d_inputs)
            next_state = state
            d_weights.index_add_(1, indices, d_batch_weights)
            yield t, state, d_state, d_batch_weights
        pbar.close()

        return d_state, d_weights
