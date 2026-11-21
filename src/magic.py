import importlib.util
import math
import functools
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Concatenate, Generator, Iterator, Self, overload

import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor

from src.data import Loader
from src.lobpcg import hessian_eigs
from src.logging import Logger, scaled_l2, total_l2
from src.objective import HVP, Objective, hvp_from_grads, metrics
from src.optimizers import OptimState, dict_zip
from src.second_order import EdgeFlowState, NewtonState, SecondOrderState, SharpnessState


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


class Stepper:
    """Context manager around a training step"""

    @contextmanager
    def attach(self, state: OptimState) -> Iterator[OptimState]:
        yield state

    def detach(self, next_state: OptimState) -> OptimState:
        return next_state.apply(lambda s: s.detach())


class ForwardMode(Stepper):
    """Attach a forward-mode AD vector before a step and detach on exit"""

    def __init__(self, tangent: OptimState):
        self.tangent = tangent

    @contextmanager
    def attach(self, state: OptimState) -> Iterator[OptimState]:
        with fwAD.dual_level():
            yield state.unflatten([fwAD.make_dual(s, tan) for s, tan in zip(state, self.tangent)])

    def detach(self, next_state: OptimState) -> OptimState:
        detached = next_state.apply(lambda s: s.detach())
        self.tangent = detached.unflatten([fwAD.unpack_dual(s).tangent.detach() for s in next_state])
        return detached


@dataclass(frozen=True, kw_only=True)
class Magic:
    """Training on the batches of `loader` with the loss `objective`,
        forward (`train`, `replay`) and backward (`backprop`: MAGIC).
    `train` and `backprop` are `traceable`: `.trace(...)` for their intermediate values
    """

    objective: Objective
    loader: Loader
    compile: bool = False

    @property
    def device(self) -> torch.device | str | None:
        return self.loader.device

    def _forward(self, state: OptimState, weights: Tensor | None) -> tuple[Callable, OptimState, bool]:
        """vmap the objective's forward if needed: `weights` [R, N] and/or `state` [R, *shape]"""
        batched = bool(self.objective.model_dims(state.params))
        if weights is not None and weights.ndim == 2 and not batched:
            state, batched = state.apply(lambda t: t.expand(len(weights), *t.shape).clone()), True
        assert not (batched and isinstance(state, SecondOrderState)), "can't vmap a second-order step"
        forward_fn = self.objective.forward
        if batched:
            vmapped = torch.func.vmap(forward_fn, in_dims=(0, None, None, None if weights is None else 0))
            forward_fn = lambda params, X, Y, weights=None: vmapped(params, X, Y, weights)
        return forward_fn, state, batched

    @traceable
    def train(
        self, init_state: OptimState, *, n_steps: int, start: int = 0, weights: Tensor | None = None,
        log: Logger | None = None, create_graph: bool = False, stepper: Stepper = Stepper(), pbar: bool = True,
    ) -> Generator[tuple[int, OptimState, dict[str, Tensor] | None], None, OptimState]:
        """Train on steps [start, start + n_steps) and return the trained state.
        `create_graph`: yield the grads with their graph (for HVPs)
        `stepper`: e.g. `ForwardMode` to propagate a tangent
        Yields (t, state_t, grads_t) for each step, and (T, state_T, None) at the end
        """
        state = init_state.apply(lambda t: t.to(self.device, non_blocking=True, copy=True))
        if weights is not None:
            weights = weights.to(self.device, non_blocking=True)
        second_order = isinstance(state, SecondOrderState)  # (its grads always have their graph)
        if weights is None and second_order:  # to avoid CentralFlow re-compilations
            weights = torch.ones(len(self.loader.dataset), device=self.device, dtype=next(iter(state.params.values())).dtype)
        forward_fn, state, batched = self._forward(state, weights)
        if self.compile and not isinstance(state, SecondOrderState):
            forward_fn = torch.compile(forward_fn)
        metrics_fn = torch.func.vmap(metrics, in_dims=(0, None)) if batched else metrics
        keep_graph = create_graph or second_order

        for t in tqdm(range(start, start + n_steps), disable=not pbar, desc="train"):
            X, Y, idx = self.loader.get_batch(t)
            forward_step = partial(forward_fn, X=X, Y=Y, weights=None if weights is None else weights[..., idx])

            with stepper.attach(state) as state:
                for p in state.params.values():
                    p.requires_grad_()
                next_state, grads, loss, out = state.step(t, forward_step, create_graph=create_graph and not second_order)
                next_state = stepper.detach(next_state)
            yield t, state if keep_graph else state.apply(lambda t: t.detach()), grads
            state = next_state

            if log is not None:
                train_metrics = {k: v.mean(-1) for k, v in metrics_fn(out.detach(), Y).items()}
                log.log(t, loss=loss.detach(), **train_metrics)

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
        models = self.objective.model_dims(init_state.params)  # () or (R,)
        replayed_states = self.replay(init_state, n_steps=n_steps, start=start, passes=passes)
        next_state = next(replayed_states).apply(lambda t: t.to(self.device))
        d_weights = torch.zeros(n_fns, *models, len(self.loader.dataset), device=self.device)  # df/dw

        pbar = tqdm(total=n_steps, desc="magic")
        for t, state in zip(reversed(range(start, start + n_steps)), replayed_states):
            pbar.update()
            X, Y, indices = self.loader.get_batch(t)
            state = state.apply(lambda t: t.detach().to(self.device, non_blocking=True).requires_grad_())

            weights = torch.ones(*models, len(indices), device=self.device, dtype=X.dtype).requires_grad_()
            forward_fn = partial(self._forward(state, weights)[0], X=X, Y=Y, weights=weights)
            *d_inputs, d_batch_weights = state.vjp(t, forward_fn, [*state, weights], d_state, next_state)
            d_state = d_state.unflatten(d_inputs)

            next_state = state
            d_weights.index_add_(-1, indices, d_batch_weights)
            yield t, state, d_state, d_batch_weights
        pbar.close()

        return d_state, d_weights


def to_device(tensors, device):
    """The nested dicts / lists / states `tensors` with every tensor on `device`"""
    if isinstance(tensors, Tensor):
        return tensors.to(device)
    if isinstance(tensors, OptimState):
        return tensors.apply(lambda t: t.to(device))
    if isinstance(tensors, dict):
        return {k: to_device(v, device) for k, v in tensors.items()}
    if isinstance(tensors, list):
        return [to_device(v, device) for v in tensors]
    return tensors


HAS_NVML = importlib.util.find_spec("pynvml") is not None  # `torch.cuda.utilization` needs it
RESUMED = ("ema", "pema", "warm", "prev_grads")  # the fields of `TrainLog` a killed training continues from


@dataclass
class TrainLog:
    """`Magic.train` with more logs (`train`): on top of its own (loss, metrics), one method per group, from the states and
    gradients it yields (`log`). Always on:
    - the norms of "params" and "grads", the step's stats (second-order states), "gpu_gb" (peak memory of the step and its logs),
      "gpu_util" (% of the last ~second the GPU was busy, if pynvml is installed)
    - "lr" if scheduled
    - `log_grad_displacement`: "grad_displacement" = ||g_t - g_{t-1}|| / ||g_{t-1}|| of the batch gradients, logged at t
    - `tangent`: "tangent": the norms of ds_t/ds_start v for a random v, propagated by forward-mode AD (`ForwardMode`)
    Optional, without second derivatives:
    - `log_emas`: "<name>/ema": the loss and metrics with the power-function EMA of the params of Karras et al. 2024 (arXiv:2312.02696:
      the weight of s_t is proportional to t^`ema_gamma`, so the EMA at step T averages its last ~1/gamma: gamma 16.97 is a
      relative std width of 5% of the training, t counted from step 0). Its first value is the params at `start` (standing for
      the steps before). `ema_decay`: also keeps the short-term EMA with this decay
    - `log_test`: "test_<name>": on `n_test` random test examples every `eval_every` steps (the whole test set at the last step),
      also at the EMAs ("test_<name>/ema", "test_<name>/ema_short")
    With second derivatives (`create_graph`):
    - `log_eigs`: "eig/i": top (i > 0) / most negative (i < 0) eigenvalues of the Hessian of the batch loss, "eigvec/i" their norms,
      every `eig_every` steps. The eigenpair a 2nd-order step computes (sharpness: the top, newton: the most negative) is logged
      at every step; LOBPCG only runs for 2 or more eigenpairs of that sign.
      "eig/threshold": the stability threshold on the same scale, `stability_threshold` / step size P (when P is a scalar: not adam)
    - `log_curvature`: "<direction>_hess/Hv" = ||H v|| / ||v|| and "<direction>_hess/vHv" = v^T H v / v^T v of the Hessian H of the
      batch loss along v = the batch gradient ("grad"), the tangent ("tangent") and params - short-term EMA ("ema"), every
      `eig_every` steps, one HVP each (vHv of the gradient: the batch sharpness)
    `checkpoint_every`: save the state to `save_dir`/checkpoints/<t>.pt every this many steps, the logs so far with their plots
    to `save_dir`/metrics.pt (as `logger.finish`), and what the logs keep between steps (`RESUMED`, the tangent) to
    `save_dir`/resume.pt, which `train` continues from if it exists
    A batched state (R models [R, *shape], first-order optimizers) logs a value [R] per model, without eigenvalues, curvature or tangent
    """

    magic: Magic
    logger: Logger
    test_loader: Loader
    n_top_eigs: int = 0
    n_neg_eigs: int = 0
    eig_every: int = 10
    curvature: bool = False
    tangent: bool = False
    eval_every: int = 100
    n_test: int | None = None
    ema_gamma: float | None = None
    ema_decay: float | None = None
    checkpoint_every: int = 0
    save_dir: Path | None = None
    # kept from step to step (`RESUMED`)
    ema: dict[str, Tensor] | None = None  # the short-term EMA of the params through s_t
    pema: dict[str, Tensor] | None = None  # the power-function EMA
    warm: dict[int, dict[str, Tensor]] = field(default_factory=dict)  # sign -> eigenvectors [k, *shape] per param, LOBPCG's warm start
    prev_grads: dict[str, Tensor] | None = None  # g_{t-1}

    def train(self, init_state: OptimState, *, n_steps: int, start: int = 0) -> OptimState:
        """Train on steps [start, start + n_steps) and return the trained state"""
        self.start = start
        self.batched = bool(self.magic.objective.model_dims(init_state.params))
        assert not (self.batched and (self.n_top_eigs or self.n_neg_eigs or self.curvature or self.tangent)), "a batched state logs no eigenvalues, curvature or tangent"
        stepped = {1: "sharpness"} if isinstance(init_state, (SharpnessState, EdgeFlowState)) \
            else {-1: "min_eig"} if isinstance(init_state, NewtonState) \
            else {}
        requested = ((1, self.n_top_eigs), (-1, self.n_neg_eigs))
        self.lobpcg = {sign: k for sign, k in requested if k > 1 or k == 1 and sign not in stepped}  # sign -> eigenpairs by LOBPCG
        self.stored = {sign: name for sign, name in stepped.items() if sign not in self.lobpcg}  # sign -> the stat of the eigenpair the step keeps (aux `eigvec`)
        stepper = ForwardMode(self._random_tangent(init_state)) if self.tangent else Stepper()
        first = start  # the step the training (re)starts from
        if self.save_dir is not None and (self.save_dir / "resume.pt").exists():
            first, init_state = self._resume(init_state, stepper)
        trace = self.magic.train.trace(
            init_state, n_steps=start + n_steps - first, start=first, log=self.logger,
            create_graph=bool(self.lobpcg) or self.curvature, stepper=stepper,
        )
        tangent = getattr(stepper, "tangent", None)  # at `state`
        for t, state, grads in trace:
            if self.checkpoint_every and t > first and grads is not None and (t - start) % self.checkpoint_every == 0:
                self._checkpoint(t, state, tangent)
            self.log(t, state, grads, tangent)
            tangent = getattr(stepper, "tangent", None)  # at the next state
        return state.apply(lambda s: s.detach().cpu())

    def _random_tangent(self, init_state: OptimState) -> OptimState:
        """Each tensor random, of `scaled_l2` norm 1"""
        g = torch.Generator().manual_seed(self.start)
        tangent = init_state.apply(lambda s: torch.randn(s.shape, generator=g, dtype=torch.float32).to(self.magic.device, s.dtype))
        return tangent.apply(lambda s: s / scaled_l2(s))

    def _checkpoint(self, t: int, state: OptimState, tangent: OptimState | None):
        assert self.save_dir is not None
        (self.save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(asdict(state.apply(lambda s: s.detach().cpu())), self.save_dir / "checkpoints" / f"{t}.pt")
        self.logger.finish(self.save_dir / "metrics.pt")
        resume = {"step": t, **{name: getattr(self, name) for name in RESUMED}, "tangent": None if tangent is None else list(tangent)}
        torch.save(to_device(resume, "cpu"), self.save_dir / "resume.pt")

    def _resume(self, init_state: OptimState, stepper: Stepper) -> tuple[int, OptimState]:
        """Continue from the last `_checkpoint`: (its step, its state)"""
        assert self.save_dir is not None
        resume = torch.load(self.save_dir / "resume.pt", weights_only=True)
        for name in RESUMED:
            setattr(self, name, to_device(resume[name], self.magic.device))
        self.logger.restore(torch.load(self.save_dir / "metrics.pt", weights_only=True))
        saved = torch.load(self.save_dir / "checkpoints" / f"{resume['step']}.pt", weights_only=True)
        state = replace(init_state, **{group: saved[group] for group in (*init_state.groups, *init_state.aux)})
        if isinstance(stepper, ForwardMode):
            stepper.tangent = to_device(state.unflatten(resume["tangent"]), self.magic.device)
        return resume["step"], state

    def log(self, t: int, state: OptimState, grads: dict[str, Tensor] | None, tangent: OptimState | None):
        """The logs of step `t`: the state before it, its gradients (None at the final state) and the tangent at `state`"""
        params = {k: p.detach() for k, p in state.params.items()}
        X, Y, _ = self.magic.loader.get_batch(t)
        if state.lr.kind != "constant" or state.lr.warmup:
            self.logger.log(t, lr=torch.tensor(state.lr(t)))
        if grads is not None:
            self.logger.log_norms(t, "params", state.params, self.batched)
            self.logger.log_norms(t, "grads", grads, self.batched)
            self.log_grad_displacement(t, grads)
        if isinstance(state, SecondOrderState) and t > self.start:  # the stats of step t - 1, which produced `state`
            self.logger.log(t - 1, **state.stats)
        self.log_emas(t, params, X, Y)
        if tangent is not None:
            self.logger.log_norms(t, "tangent", tangent)
        self.log_eigs(t, state, grads, params)
        self.log_curvature(t, state, grads, params, tangent)
        self.log_test(t, grads is None, params)
        if grads is not None and X.is_cuda:
            self.logger.log(t, gpu_gb=torch.tensor(torch.cuda.max_memory_allocated() / 2**30))
            torch.cuda.reset_peak_memory_stats()
            if HAS_NVML:
                self.logger.log(t, gpu_util=torch.tensor(float(torch.cuda.utilization(X.device))))

    def log_grad_displacement(self, t: int, grads: dict[str, Tensor]):
        grads = {k: g.detach() for k, g in grads.items()}
        if self.prev_grads is not None:
            grad_diff = {k: g - g_prev for k, (g, g_prev) in dict_zip(grads, self.prev_grads).items()}
            self.logger.log(t, grad_displacement=total_l2(grad_diff.values(), self.batched) / total_l2(self.prev_grads.values(), self.batched))
        self.prev_grads = grads

    def log_emas(self, t: int, params: dict[str, Tensor], X: Tensor, Y: Tensor):
        if self.ema_decay is not None:
            self.ema = params if self.ema is None else {k: e + (1 - self.ema_decay) * (p - e) for k, (e, p) in dict_zip(self.ema, params).items()}
        if self.ema_gamma is not None:
            beta = (1 - 1 / (t + 1)) ** (self.ema_gamma + 1)  # the step t is the (t + 1)th state
            self.pema = params if self.pema is None else {k: e + (1 - beta) * (p - e) for k, (e, p) in dict_zip(self.pema, params).items()}
            forward, batch_metrics_fn = self.magic.objective.forward, metrics
            if self.batched:
                forward, batch_metrics_fn = torch.func.vmap(forward, in_dims=(0, None, None)), torch.func.vmap(metrics, in_dims=(0, None))
            with torch.no_grad():
                loss, out = forward(self.pema, X, Y)
            batch_metrics = {"loss": loss, **{k: m.mean(-1) for k, m in batch_metrics_fn(out, Y).items()}}
            self.logger.log(t, **{f"{k}/ema": m for k, m in batch_metrics.items()})

    def log_eigs(self, t: int, state: OptimState, grads: dict[str, Tensor] | None, params: dict[str, Tensor]):
        if self.stored and t > self.start:  # the eigenpair step t - 1 computed at s_{t-1}
            assert isinstance(state, SecondOrderState)
            for sign, name in self.stored.items():
                self._log_eigenpairs(t - 1, [sign], state.stats[name][None], {k: u[None] for k, u in state.eigvec.items()})
            self._log_threshold(t, state)
        if self.lobpcg and self._eig_step(t, grads) and t > self.start:
            hvp_fn = hvp_from_grads(grads, state.params)
            for sign, k in self.lobpcg.items():
                if sign not in self.warm:  # random start
                    g = torch.Generator().manual_seed(t)
                    self.warm[sign] = {name: torch.randn(k, *p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype) for name, p in params.items()}
                eigvals, self.warm[sign], n_iter = hessian_eigs(hvp_fn, self.warm[sign], sign)
                self.logger.log(t, **{f"lobpcg_iters/{sign}": torch.tensor(n_iter)})
                self._log_eigenpairs(t, [sign * (i + 1) for i in range(k)], eigvals, self.warm[sign])
            if not self.stored:
                self._log_threshold(t, state)

    def log_curvature(
        self, t: int, state: OptimState, grads: dict[str, Tensor] | None, params: dict[str, Tensor], tangent: OptimState | None
    ):
        if not (self.curvature and self._eig_step(t, grads)):
            return
        hvp_fn = hvp_from_grads(grads, state.params)
        directions = {"grad": {k: g.detach() for k, g in grads.items()},
                      "tangent": None if tangent is None else tangent.params,
                      "ema": None if self.ema is None or t == self.start else {k: p - e for k, (p, e) in dict_zip(params, self.ema).items()}}
        for name, direction in directions.items():
            if direction is not None:
                self._log_direction(t, name, direction, hvp_fn)
        if not self.lobpcg and not self.stored:
            self._log_threshold(t, state)

    def log_test(self, t: int, last: bool, params: dict[str, Tensor]):
        if not (self.eval_every and ((t - self.start) % self.eval_every == 0 or last)):
            return
        test_ds = self.test_loader.dataset
        indices = torch.arange(len(test_ds)) if self.n_test is None or last else test_ds.subset(self.n_test, index=t)
        for p, suffix in ((params, ""), (self.pema, "/ema"), (self.ema, "/ema_short")):
            if p is not None:
                loss, mean_metrics = self.magic.objective.eval_loss(p, self.test_loader.iter_batches(indices), self.batched)
                self.logger.log(t, **{f"test_{k}{suffix}": m for k, m in {"loss": loss, **mean_metrics}.items()})

    def _eig_step(self, t: int, grads: dict[str, Tensor] | None) -> bool:
        return grads is not None and (t - self.start) % self.eig_every == 0

    def _log_eigenpairs(self, t: int, ids: list[int], eigvals: Tensor, eigvecs: dict[str, Tensor]):
        """Log "eig/i" and the norms "eigvec/i" of the eigenpairs `ids` (eigvecs: [k, *shape] per param)"""
        self.logger.log(t, **{f"eig/{i}": lam for i, lam in zip(ids, eigvals.unbind())})
        for j, i in enumerate(ids):
            self.logger.log_norms(t, f"eigvec/{i}", {name: vec[j] for name, vec in eigvecs.items()})

    def _log_direction(self, t: int, name: str, direction: dict[str, Tensor], hvp_fn: HVP):
        """Log "<name>_hess/Hv" = ||H v|| / ||v|| and "<name>_hess/vHv" = v^T H v / v^T v (one HVP)"""
        Hv = hvp_fn(direction)
        vv = torch.stack([u.pow(2).sum() for u in direction.values()]).sum()
        vHv = torch.stack([(u * hu).sum() for u, hu in dict_zip(direction, Hv).values()]).sum()
        HvHv = torch.stack([hu.pow(2).sum() for hu in Hv.values()]).sum()
        self.logger.log(t, **{f"{name}_hess/Hv": (HvHv / vv).sqrt(), f"{name}_hess/vHv": vHv / vv})

    def _log_threshold(self, t: int, state: OptimState):
        """"eig/threshold" = stability threshold / step size, if the step size is a scalar. A second-order state's `precond`
        uses the stats of the step that produced it (step t - 1, at s_{t-1}): logged at t - 1, like its eigenpair"""
        if isinstance(state, SecondOrderState):
            if t == self.start:
                return  # (no stats yet)
            t -= 1
        P = next(iter(state.precond(t).values()))
        if not isinstance(P, Tensor) or P.ndim == 0:
            threshold = state.stability_threshold / P
            self.logger.log(t, **{"eig/threshold": torch.as_tensor(threshold, dtype=torch.get_default_dtype())})


def backprop_logged(
    magic: Magic, init_state: OptimState, d_state: OptimState, eval_sets: list[tuple[Tensor, Tensor]], logger: Logger, **kwargs
) -> tuple[OptimState, Tensor]:
    """`Magic.backprop` logging, per step and eval set i: "f/i" = f_i(s_t), |df_i/ds_t| ("d_state/i", over all the state's tensors)
    and |df_i/dw_t| ("d_weights/i") (a batched `init_state` of R models: [R] each)"""
    eval_sets = to_device([list(eval_set) for eval_set in eval_sets], magic.device)
    batched = bool(magic.objective.model_dims(init_state.params))
    forward = torch.func.vmap(magic.objective.forward, in_dims=(0, None, None)) if batched else magic.objective.forward
    trace = magic.backprop.trace(init_state, d_state, **kwargs)
    while True:
        try:
            t, state, d_state_t, d_batch_weights = next(trace)
        except StopIteration as stop:
            return stop.value
        params = {k: p.detach() for k, p in state.params.items()}
        d_state_norms = total_l2(d_state_t, 1 + batched)  # [#fns, (R,)]
        for i, (X_i, Y_i) in enumerate(eval_sets):
            with torch.no_grad():
                logger.log(t, **{f"f/{i}": forward(params, X_i, Y_i)[0], f"d_state/{i}": d_state_norms[i], f"d_weights/{i}": d_batch_weights[i].double().norm(dim=-1)})
