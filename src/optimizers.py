"""Optimizer states: the model parameters + the optimizer's tensors, and the step on them (`OptimState.step`, a pure function
of the state and the batch loss, so that MAGIC can differentiate through it), with the learning rate schedule."""

import math
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Callable, ClassVar, Iterator, Iterable, Mapping, NamedTuple, Self, Sequence

import torch
from torch import Tensor

from src.minres import minres
from src.utils import HVP, dict_zip, hvp_from_forward, hvp_from_grads
from src.lobpcg import flatten, hessian_eigs, unflatten
from src.sdcp import sdcp

Forward = Callable[[dict[str, Tensor]], tuple[Tensor, Tensor]]  # params -> (batch loss [(R,)], outputs [(R,) B, C])
StepOutput = tuple[dict[str, Tensor], Tensor, Tensor]  # (grads, loss, outputs)


class LossGraph(NamedTuple):
    """The gradients' graph of a step's loss on one device"""

    params: dict[str, Tensor]
    hvp_fn: HVP
    inputs: Sequence[Tensor]  # the `vjp` inputs on the device


@dataclass(frozen=True)
class LRSchedule:
    """The learning rate at step t (`schedule(t)`): linear warmup to `lr` over `warmup` steps, then
    - "constant": lr
    - "cosine": from lr at step `warmup` to FINAL_FRACTION * lr at step `n_steps`
    """

    lr: float
    kind: str = "constant"
    warmup: int = 0
    n_steps: int = 0
    FINAL_FRACTION: ClassVar[float] = 0.1

    def __call__(self, t: int) -> float:
        if t < self.warmup:
            return self.lr * (t + 1) / self.warmup
        if self.kind == "constant":
            return self.lr
        assert self.kind == "cosine", self.kind
        progress = min(1.0, (t - self.warmup) / max(1, self.n_steps - self.warmup))
        cos = (1 + math.cos(math.pi * progress)) / 2
        return self.lr * (self.FINAL_FRACTION + (1 - self.FINAL_FRACTION) * cos)


@dataclass(kw_only=True)
class OptimState:
    """All model + optimizer state tensors, organized into dict[str, Tensor] objects:
    - `groups`: the differentiable state (what MAGIC backpropagates through)
    - `aux`: non-differentiable state, carried along detached
    """

    groups: ClassVar[tuple[str, ...]] = ("params",)
    aux: ClassVar[tuple[str, ...]] = ()

    params: dict[str, Tensor]  # model parameters
    lr: LRSchedule
    wd: float = 0.0  # weight decay

    def __post_init__(self):
        if isinstance(self.lr, dict):  # saved by `asdict`
            self.lr = LRSchedule(**self.lr)

    def __iter__(self) -> Iterator[Tensor]:
        """All differentiable state tensors"""
        return (t for group in self.groups for t in getattr(self, group).values())

    def unflatten(self, tensors: Iterable[Tensor]) -> Self:
        """Inverse of iterating: a state like this one holding `tensors` instead"""
        it = iter(tensors)
        new_state = {
            group: {k: next(it) for k in getattr(self, group)} for group in self.groups
        }
        return replace(self, **new_state)

    def apply(self, fn: Callable[[Tensor], Tensor]) -> Self:
        """A copy of the state with each tensor (including aux, kept detached) replaced by `fn(tensor)`"""
        new_state: dict[str, dict[str, Tensor]] = {}
        for group in (*self.groups, *self.aux):
            new_state[group] = {k: fn(t) for k, t in getattr(self, group).items()}
            if group in self.aux:
                new_state[group] = {k: t.detach() for k, t in new_state[group].items()}
        return replace(self, **new_state)

    def step(self, t: int, forward_fn: Forward, create_graph: bool = False) -> tuple[Self, *StepOutput]:
        """Step `t` on the loss of `forward_fn` (one loss per model if the state is batched).
        `create_graph`: make the step differentiable"""
        loss, out = forward_fn(self.params)  # () or (#models,)
        grads = torch.autograd.grad(loss.sum(), self.params, create_graph=create_graph)
        return self._update(t, grads), grads, loss, out

    def _update(self, t: int, grads: dict[str, Tensor]) -> Self:
        """The state after step `t` with the gradients `grads`"""
        raise NotImplementedError

    def vjp(
        self, t: int, forward_fn: Forward, inputs: Sequence[Tensor], cotangents: "OptimState"
    ) -> Sequence[Tensor]:
        """Backward of `step`: d(next_state)/d(inputs) at `cotangents` (batched)"""
        next_state = self.step(t, forward_fn, create_graph=True)[0]
        return torch.autograd.grad(list(next_state), inputs, list(cotangents), is_grads_batched=True)

    def precond(self, t: int) -> Mapping[str, Tensor | float]:
        """Per-element step size P. The update is P * grad"""
        return {k: self.lr(t) for k in self.params}

    @property
    def stability_threshold(self) -> float:
        """The EoS condition is `max_eig(sqrt(P) H sqrt(P)) = stability_threshold`"""
        raise NotImplementedError


@dataclass(kw_only=True)
class SGDState(OptimState):
    def _update(self, t: int, grads: dict[str, Tensor]) -> Self:
        params = {
            k: p - self.lr(t) * (g + self.wd * p)
            for k, (p, g) in dict_zip(self.params, grads).items()
        }
        return replace(self, params=params)

    def gn_vjp(self, t: int, cotangent: OptimState, gnvp_fn: HVP) -> tuple[Self, dict[str, Tensor]]:
        """Backward of the SGD step with the Gauss-Newton matrix as the Hessian: (d/d(state), d/d(grads)) at `cotangent` (batched)"""
        lr = self.lr(t)
        d_grads = {k: -lr * d for k, d in cotangent.params.items()}
        d_params = {
            k: (1 - lr * self.wd) * d + Gd
            for k, (d, Gd) in dict_zip(cotangent.params, gnvp_fn(d_grads)).items()
        }
        return replace(cotangent, params=d_params), d_grads

    @property
    def stability_threshold(self) -> float:
        return 2.0


@dataclass(kw_only=True)
class SGDMomentumState(SGDState):
    groups = ("params", "buffer")
    momentum: float = 0.9
    buffer: dict[str, Tensor] = field(default_factory=dict)  # None -> zeros

    def __post_init__(self):
        super().__post_init__()
        if not self.buffer:
            self.buffer = {k: torch.zeros_like(p) for k, p in self.params.items()}

    def _update(self, t: int, grads: dict[str, Tensor]) -> Self:
        buffer = {
            k: self.momentum * buf + g + self.wd * p
            for k, (p, buf, g) in dict_zip(self.params, self.buffer, grads).items()
        }
        params = {
            k: p - self.lr(t) * buf for k, (p, buf) in dict_zip(self.params, buffer).items()
        }
        return replace(self, params=params, buffer=buffer)

    @property
    def stability_threshold(self) -> float:
        return 2 * (1 + self.momentum)


@dataclass(kw_only=True)
class AdamState(OptimState):
    """Adam as in torch.optim.Adam (L2 weight decay), with the μP learning rates `lrs`."""

    groups = ("params", "first_moment", "second_moment")
    lr_mults: dict[str, float] = field(default_factory=dict)  # per-tensor lr multipliers (μP: 1/sqrt(numel))
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    first_moment: dict[str, Tensor] = field(default_factory=dict)  # None -> zeros
    second_moment: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not self.lr_mults:
            self.lr_mults = {k: p.numel() ** -0.5 for k, p in self.params.items()}
        if not self.first_moment:
            self.first_moment = {k: torch.zeros_like(p) for k, p in self.params.items()}
        if not self.second_moment:
            self.second_moment = {k: torch.zeros_like(p) for k, p in self.params.items()}

    def _step_sizes(self, second_moment: dict[str, Tensor], t: int) -> dict[str, Tensor]:
        """Per-element step size of step t The (t + 1)-th, for the bias correction"""
        beta1, beta2, n_steps, lr = *self.betas, t + 1, self.lr(t)
        bias_corrected = {k: v / (1 - beta2**n_steps) for k, v in second_moment.items()}
        return {
            k: lr * self.lr_mults[k] / (1 - beta1**n_steps) / (v**0.5 + self.eps)
            for k, v in bias_corrected.items()
        }

    def _update(self, t: int, grads: dict[str, Tensor]) -> Self:
        beta1, beta2 = self.betas
        reg_grads = {
            k: g + self.wd * p for k, (g, p) in dict_zip(grads, self.params).items()
        }
        first_moment = {
            k: beta1 * m + (1 - beta1) * g
            for k, (m, g) in dict_zip(self.first_moment, reg_grads).items()
        }
        second_moment = {
            k: beta2 * v + (1 - beta2) * g * g
            for k, (v, g) in dict_zip(self.second_moment, reg_grads).items()
        }
        step_sizes = self._step_sizes(second_moment, t)
        params = {
            k: p - size * m
            for k, (p, m, size) in dict_zip(self.params, first_moment, step_sizes).items()
        }
        return replace(self, params=params, first_moment=first_moment, second_moment=second_moment)

    def precond(self, t: int) -> dict[str, Tensor]:
        return self._step_sizes(self.second_moment, t)

    @property
    def stability_threshold(self) -> float:
        """Cohen et al. 2022, "Adaptive gradient methods at the edge of stability" """
        return 2 * (1 + self.betas[0]) / (1 - self.betas[0])


@dataclass(kw_only=True)
class HeunState(OptimState):
    """2-stage Runge-Kutta on the gradient flow (+ eps * 1-stage loss)"""

    eps: float = 0.0  # weight of the 1-stage loss

    def step(self, t: int, forward_fn: Forward, create_graph: bool = False) -> tuple[Self, *StepOutput]:
        """The grads, loss and outputs are at `params` (the first stage)"""
        loss, out = forward_fn(self.params)
        grads = torch.autograd.grad(loss.sum(), self.params, create_graph=create_graph)

        trial_params = {k: p - self.lr(t) * (g + self.wd * p) for k, (p, g) in dict_zip(self.params, grads).items()}
        trial_loss = forward_fn(trial_params)[0]
        trial_grads = torch.autograd.grad(trial_loss.sum(), trial_params, create_graph=create_graph)

        params = {
            k: p - self.lr(t) * 0.5 * ((1 + self.eps) * (g + self.wd * p) + (1 - self.eps) * (trial_g + self.wd * trial_p))
            for k, (p, g, trial_p, trial_g) in dict_zip(self.params, grads, trial_params, trial_grads).items()
        }
        return replace(self, params=params), grads, loss, out

    @property
    def stability_threshold(self) -> float:
        return 2.0


@dataclass(kw_only=True)
class SecondOrderState(OptimState):
    aux = ("stats",)
    stats: dict[str, Tensor] = field(default_factory=dict)

    def step(self, t: int, forward_fn: Forward, create_graph: bool = False) -> tuple[Self, *StepOutput]:
        """The grads always have their graph (for the HVPs); `create_graph` is of the HVPs"""
        loss, out = forward_fn(self.params)
        grads = torch.autograd.grad(loss, self.params, create_graph=True)
        return self._update(t, grads, create_graph), grads, loss, out

    def _update(self, t: int, grads: dict[str, Tensor], create_graph: bool = False) -> Self:
        raise NotImplementedError

    @property
    def stability_threshold(self) -> float:
        return 2.0


@dataclass(kw_only=True)
class SharpnessState(SecondOrderState):
    """SGD with step size `lr * 2 / sharpness`.
    Sharpness is computed with its graph, allowing backprop through a step"""

    aux = ("eigvec", "stats")
    eigvec: dict[str, Tensor] = field(default_factory=dict)  # top eigenvector (warm start)

    def __post_init__(self):
        super().__post_init__()
        if not self.eigvec:
            g = torch.Generator().manual_seed(0)
            self.eigvec = {
                k: torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype)
                for k, p in self.params.items()
            }
        if not self.stats:
            p = next(iter(self.params.values()))
            nan = torch.tensor(float("nan"), dtype=p.dtype, device=p.device)
            self.stats = {"sharpness": nan, "eig_iters": torch.zeros_like(nan)}

    def _update(self, t: int, grads: dict[str, Tensor], create_graph: bool = False) -> Self:
        hvp_fn = hvp_from_grads(grads, self.params)
        sharpness, eigvec, eig_iters = hessian_eigs(hvp_fn, self.eigvec, batched=False)
        if create_graph:
            H_eigvec = hvp_fn(eigvec, create_graph=True)
            sharpness = torch.stack(  # S = v^T H v
                [(v * Hv).sum() for v, Hv in dict_zip(eigvec, H_eigvec).values()]
            ).sum()
        step_size = self.lr(t) * 2 / sharpness
        params = {
            k: p - step_size * (g + self.wd * p)
            for k, (p, g) in dict_zip(self.params, grads).items()
        }
        stats = {"sharpness": sharpness.detach(), "eig_iters": torch.full_like(sharpness.detach(), eig_iters)}
        return replace(self, params=params, eigvec=eigvec, stats=stats)

    def precond(self, t: int) -> Mapping[str, Tensor | float]:
        """The step size with the last step's sharpness (0 before the first step)"""
        return {k: self.lr(t) * 2 / self.stats["sharpness"] for k in self.params}


@dataclass(kw_only=True)
class EdgeFlowState(SecondOrderState):
    """Edge Flow (Marion 2026, Algorithm 1): `lr` is the modeled GD learning rate eta
    update is 1/2 * (grad(params + eps * eigvec) + grad(params - eps * eigvec))
    `eigvec` (a power iteration) and `eps` (grows iff sharpness > 2 / eta)"""

    SUBSTEPS: ClassVar[int] = 4
    MIN_EPS: ClassVar[float] = 1e-4

    groups = ("params", "eigvec", "eps")
    eigvec: dict[str, Tensor] = field(default_factory=dict)  # None -> random unit
    eps: dict[str, Tensor] = field(default_factory=dict)  # {"eps": []}; None -> MIN_EPS

    def __post_init__(self):
        super().__post_init__()
        p = next(iter(self.params.values()))
        if not self.eigvec:
            g = torch.Generator().manual_seed(0)
            noise = {k: torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype) for k, p in self.params.items()}
            norm = torch.sqrt(sum((v * v).sum() for v in noise.values()))
            self.eigvec = {k: v / norm for k, v in noise.items()}
        if not self.eps:
            self.eps = {"eps": torch.full((), self.MIN_EPS, dtype=p.dtype, device=p.device)}
        if not self.stats:
            nan = torch.tensor(float("nan"), dtype=p.dtype, device=p.device)
            self.stats = {"sharpness": nan, "eps": self.eps["eps"].detach(), "center_loss": nan}

    def step(self, t: int, forward_fn: Forward, create_graph: bool = False) -> tuple[Self, *StepOutput]:
        """The loss and grads are averaged over the endpoints `params` +- eps * eigvec; the outputs are at `params`"""
        eps = self.eps["eps"]
        left, right = [
            {k: p + sign * eps * v for k, (p, v) in dict_zip(self.params, self.eigvec).items()} for sign in (1, -1)
        ]
        loss = (forward_fn(left)[0] + forward_fn(right)[0]) / 2
        center_loss, out = forward_fn(self.params)
        grads = torch.autograd.grad(loss, self.params, create_graph=create_graph)
        center_grads = torch.autograd.grad(center_loss, self.params, create_graph=True)

        H_eigvec = hvp_from_grads(center_grads, self.params)(self.eigvec, create_graph=create_graph)
        sharpness = torch.stack([hv.pow(2).sum() for hv in H_eigvec.values()]).sum().sqrt()  # |H eigvec|: keeps eigvec unit
        rho = self.lr(t) / self.SUBSTEPS  # Edge-Flow step size
        params = {k: p - rho * (g + self.wd * p) for k, (p, g) in dict_zip(self.params, grads).items()}
        eigvec = {k: hv / sharpness for k, hv in H_eigvec.items()}
        eps = (eps * (1 + rho * (sharpness - 2 / self.lr(t)))).clamp_min(self.MIN_EPS)

        stats = {"sharpness": sharpness.detach(), "eps": eps.detach(), "center_loss": center_loss.detach()}
        new_state = replace(self, params=params, eigvec=eigvec, eps={"eps": eps}, stats=stats)
        return new_state, center_grads, loss, out

    def precond(self, t: int) -> Mapping[str, Tensor | float]:
        return {k: self.lr(t) / self.SUBSTEPS for k in self.params}

    @property
    def stability_threshold(self) -> float:
        """2 / eta on the scale of the step rho"""
        return 2 / self.SUBSTEPS


@dataclass(kw_only=True)
class CentralFlowState(SecondOrderState):
    """Central flow of GD with learning rate `lr` (Cohen et al. 2024, arXiv:2410.24206), SUBSTEPS Euler steps per GD step:
        params <- params - lr / SUBSTEPS * (grads + 1/2 * d<Sigma, H>/d(params))
    Sigma = U X U^T, the covariance of GD's oscillations: U the critical eigenvectors (eigenvalues above THRESHOLD / lr),
    X the `sdcp` solution that keeps U^T H U <= 2 / lr after the step.
    Backward: `vjp`, as `step` with `create_graph` holds U constant (far off); it adds dU by one MINRES solve per critical eigenvector"""

    SUBSTEPS: ClassVar[int] = 4
    N_EIGS: ClassVar[int] = 5  # at most this many critical eigenvalues (tracked: one more than the last step's, at least 4)
    THRESHOLD: ClassVar[float] = 1.5  # an eigenvalue is critical above THRESHOLD / lr
    EIG_RTOL: ClassVar[float] = 1e-2  # of LOBPCG
    VJP_RTOL: ClassVar[float] = 1e-4  # of MINRES

    aux = ("eigvecs", "stats")
    eigvecs: dict[str, Tensor] = field(default_factory=dict)  # the tracked eigenvectors (warm start); None -> random
    compile: bool = False  # torch.compile the HVPs of LOBPCG
    devices: tuple[str, ...] = ()  # helper devices of `vjp`: the pairs of critical eigenvectors are split among them and the state's own

    def __post_init__(self):
        super().__post_init__()
        if not self.eigvecs:
            g = torch.Generator().manual_seed(0)
            self.eigvecs = {
                k: torch.randn(4, *p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype)
                for k, p in self.params.items()
            }
        if self.compile:  # LOBPCG's HVPs are compiled for k and 3k vectors, k = 4, ..., N_EIGS + 1
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 2 * (self.N_EIGS - 2))

    def _track(self, eigvecs: dict[str, Tensor], n_critical: int) -> dict[str, Tensor]:
        """The next step's warm start: the top n_critical + 1 of `eigvecs`, the new ones random"""
        n_old = len(next(iter(eigvecs.values())))
        n_new = min(max(n_critical + 1, 4), self.N_EIGS + 1) - n_old
        if n_new <= 0:
            return {k: v[: n_old + n_new] for k, v in eigvecs.items()}
        g = torch.Generator().manual_seed(0)
        return {
            k: torch.cat([v, torch.randn(n_new, *v.shape[1:], generator=g, dtype=torch.float32).to(v.device, v.dtype)])
            for k, v in eigvecs.items()
        }

    def _top_eigs(self, t: int, forward_fn: Forward) -> tuple[Tensor, dict[str, Tensor], int, int]:
        """(tracked eigenvalues, their eigenvectors, LOBPCG iterations, number of critical eigenvalues)"""
        with torch.no_grad():  # (`forward_fn` may hold tensors that require grad)
            hvp_fn = hvp_from_forward(forward_fn, self.params, self.compile)
            eigvals, eigvecs, eig_iters = hessian_eigs(hvp_fn, self.eigvecs, rtol=self.EIG_RTOL)
        n_critical = int((eigvals[: self.N_EIGS] + self.wd > self.THRESHOLD / self.lr(t)).sum())
        return eigvals, eigvecs, eig_iters, n_critical

    def _loss_graphs(
        self, forward_fn: Forward, grads: dict[str, Tensor], devices: Sequence[str], inputs: Sequence[Tensor] = ()
    ) -> list[LossGraph]:
        """The graph of `grads`, and its copies on `devices` (queued there: computed while this device goes on)"""
        graphs = [LossGraph(self.params, hvp_from_grads(grads, self.params), inputs)]
        for device in devices:
            assert isinstance(forward_fn, partial), "helper devices: the batch is moved from the partial's arguments"
            copies = {id(x): x.detach().to(device).requires_grad_() for x in (*self.params.values(), *inputs)}
            move = lambda x: copies.get(id(x), x.detach().to(device)) if isinstance(x, Tensor) else x
            params = {k: move(p) for k, p in self.params.items()}
            # TODO: this is kinda hacky, fine for now though
            device_forward = partial(forward_fn.func, *map(move, forward_fn.args), **{k: move(x) for k, x in forward_fn.keywords.items()})
            device_grads = torch.autograd.grad(device_forward(params)[0], params, create_graph=True)
            graphs.append(LossGraph(params, hvp_from_grads(device_grads, params), [copies[id(x)] for x in inputs]))
        return graphs

    @staticmethod
    def _curvature(
        graph: LossGraph, critical: dict[str, Tensor], i: int, j: int, create_graph: bool = False
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """u_i^T H u_j [] and its gradient {name: [*shape]}, on the graph's device"""
        H_eigvec = graph.hvp_fn({k: u[i] for k, u in critical.items()}, create_graph=True)
        other = [critical[k][j] for k in H_eigvec]
        curvature = torch.stack([(u * Hu).sum() for u, Hu in zip(other, H_eigvec.values())]).sum()
        curvature_grads = torch.autograd.grad(
            list(H_eigvec.values()), graph.params, other, retain_graph=True, create_graph=create_graph
        )
        return (curvature, curvature_grads) if create_graph else (curvature.detach(), curvature_grads)

    def _curvatures(
        self, graphs: Sequence[LossGraph], critical: dict[str, Tensor], create_graph: bool = False
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """`_curvature` of the pairs i <= j in row order, split among `graphs`: [#pairs], {name: [#pairs, *shape]}"""
        n_critical = len(next(iter(critical.values())))
        pairs = [(i, j) for i in range(n_critical) for j in range(i, n_critical)]
        results: dict[int, tuple[Tensor, dict[str, Tensor]]] = {}
        for g, graph in reversed(list(enumerate(graphs))):  # this device last: nothing is moved once it is busy
            device_critical = {k: u.to(graph.params[k].device) for k, u in critical.items()}
            for pair in range(g, len(pairs), len(graphs)):
                results[pair] = self._curvature(graph, device_critical, *pairs[pair], create_graph)
        device = next(iter(critical.values())).device
        curvatures = torch.stack([results[pair][0].to(device) for pair in range(len(pairs))])
        curvature_grads = {k: torch.stack([results[pair][1][k].to(device) for pair in range(len(pairs))]) for k in critical}
        return curvatures, curvature_grads

    def _batched_curvatures(self, hvp_fn: HVP, critical: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """`_curvatures` on this device alone: one batched backward per eigenvector i, over its pairs j >= i"""
        curvatures, curvature_grads = [], []
        for i in range(len(next(iter(critical.values())))):
            H_eigvec = hvp_fn({k: u[i] for k, u in critical.items()}, create_graph=True)
            others = [critical[k][i:] for k in H_eigvec]
            curvatures.append(torch.stack([(u * Hu.detach()).flatten(1).sum(1) for u, Hu in zip(others, H_eigvec.values())]).sum(0))
            curvature_grads.append(torch.autograd.grad(list(H_eigvec.values()), self.params, others, retain_graph=True, is_grads_batched=True))
        return torch.cat(curvatures), {k: torch.cat([grads[k] for grads in curvature_grads]) for k in critical}

    def _flow_update(
        self, t: int, grads: dict[str, Tensor], curvatures: Tensor, curvature_grads: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor]:
        """The Euler step from u_i^T H u_j [#pairs] and their gradients, for the pairs i <= j in row order -> (params, X [k, k])"""
        lr, n_critical = self.lr(t), math.isqrt(2 * len(curvatures))
        step_size = lr / self.SUBSTEPS
        reg_grads = {k: g + self.wd * p for k, (p, g) in dict_zip(self.params, grads).items()}
        # along the step, d(curvatures) = -step_size * (drift + gram @ (the pairs of X, off-diagonal twice) / 2)
        drift = torch.stack([(dc * g).flatten(1).sum(1) for dc, g in dict_zip(curvature_grads, reg_grads).values()]).sum(0)
        gram = torch.stack([dc.flatten(1) @ dc.flatten(1).T for dc in curvature_grads.values()]).sum(0)
        rows, cols = torch.triu_indices(n_critical, n_critical, device=curvatures.device)
        pair = torch.zeros(n_critical, n_critical, dtype=torch.long, device=rows.device)  # [i, j] -> the pair's index
        pair[rows, cols] = pair[cols, rows] = torch.arange(len(rows), device=rows.device)

        edge = torch.eye(n_critical).to(curvatures) * (2 / lr - self.wd)
        X = sdcp(edge - (curvatures - step_size * drift)[pair], step_size / 2 * gram[pair][:, :, pair])
        pair_weights = X[rows, cols] * (2 - (rows == cols).to(X))
        params = {
            k: p - step_size * (g + torch.tensordot(pair_weights, dc, 1) / 2)
            for k, (p, g, dc) in dict_zip(self.params, reg_grads, curvature_grads).items()
        }
        return params, X

    def _gradient_flow_step(self, t: int, grads: dict[str, Tensor]) -> dict[str, Tensor]:
        """The Euler step without critical eigenvectors"""
        step_size = self.lr(t) / self.SUBSTEPS
        return {k: p - step_size * (g + self.wd * p) for k, (p, g) in dict_zip(self.params, grads).items()}

    def step(self, t: int, forward_fn: Forward, create_graph: bool = False) -> tuple[Self, *StepOutput]:
        """On this device alone (the helper `devices` are for `vjp`)"""
        assert not create_graph
        loss, out = forward_fn(self.params)
        grads = torch.autograd.grad(loss, self.params, create_graph=True)
        eigvals, eigvecs, eig_iters, n_critical = self._top_eigs(t, forward_fn)
        if n_critical == 0:
            params, X = self._gradient_flow_step(t, grads), torch.zeros(0, 0)
        else:
            critical = {k: v[:n_critical] for k, v in eigvecs.items()}
            params, X = self._flow_update(t, grads, *self._batched_curvatures(hvp_from_grads(grads, self.params), critical))
        stats = {f"eig/{i + 1}": eig for i, eig in enumerate(eigvals)} | {
            "eig/threshold": torch.full_like(eigvals[0], 2 / self.lr(t) - self.wd),
            "eig_iters": torch.full_like(eigvals[0], eig_iters), "n_critical": torch.full_like(eigvals[0], n_critical),
            "oscillation": X.detach().trace().to(eigvals),  # tr(Sigma): GD's mean loss is ~ loss + oscillation / lr
        }
        return replace(self, params=params, eigvecs=self._track(eigvecs, n_critical), stats=stats), grads, loss, out

    # TODO: before a full 8xH100 run, check if work is being queued (or switch to one thread per GPU)
    def vjp(
        self, t: int, forward_fn: Forward, inputs: Sequence[Tensor], cotangents: OptimState
    ) -> Sequence[Tensor]:
        """In two stages, so that only one pair's third-derivative graph is alive at a time: back to the curvatures and their
        gradients (as leaves), then through each pair's own graph.
        The step depends on U through span(U) only: with Y = d(next params)/dU at `cotangents`, the derivative of U
        adds the gradient of sum_i z_i^T H u_i at constant z, u, where (eig_i - H) z_i = y_i on the complement of span(U)"""
        grads = torch.autograd.grad(forward_fn(self.params)[0], self.params, create_graph=True)
        graphs = self._loss_graphs(forward_fn, grads, self.devices, inputs)
        hvp_fn = graphs[0].hvp_fn
        eigvals, eigvecs, _, n_critical = self._top_eigs(t, forward_fn)
        critical = {k: v[:n_critical] for k, v in eigvecs.items()}
        if n_critical == 0:
            params = self._gradient_flow_step(t, grads)
            return torch.autograd.grad(list(params.values()), inputs, list(cotangents), is_grads_batched=True)

        pairs = [(i, j) for i in range(n_critical) for j in range(i, n_critical)]
        curvatures, curvature_grads = self._curvatures(graphs, critical)
        curvatures, curvature_grads = curvatures.requires_grad_(), {k: d.requires_grad_() for k, d in curvature_grads.items()}
        params = self._flow_update(t, grads, curvatures, curvature_grads)[0]
        d_inputs = list(torch.autograd.grad(
            list(params.values()), [*inputs, curvatures, *curvature_grads.values()], list(cotangents),
            is_grads_batched=True, retain_graph=True, materialize_grads=True,
        ))
        d_curvature_grads, d_curvatures = d_inputs[len(inputs) + 1 :], d_inputs[len(inputs)]  # [#fns, #pairs, *shape] each, [#fns, #pairs]
        del d_inputs[len(inputs) :], params

        d_critical = [torch.zeros_like(d[:, :n_critical]) for d in d_curvature_grads]  # [#fns, k, *shape] each
        d_graphs = []  # per graph: the sum over its pairs of d/d(its inputs, its critical eigenvectors), on its device
        for g, graph in reversed(list(enumerate(graphs))):  # this device last: nothing is moved once it is busy
            device = next(iter(graph.params.values())).device
            leaves = {k: u.to(device, copy=True).requires_grad_() for k, u in critical.items()}
            shard = list(range(g, len(pairs), len(graphs)))
            cotangent_curvatures = d_curvatures[:, shard].to(device)
            cotangent_grads = [d[:, shard].to(device) for d in d_curvature_grads]
            d_graph = None
            for n, pair in enumerate(shard):  # one pair and one fn at a time: the backward through a third-derivative graph is the memory constraint
                curvature, pair_grads = self._curvature(graph, leaves, *pairs[pair], create_graph=True)
                d_pair = [
                    torch.autograd.grad(
                        [curvature, *pair_grads.values()], [*graph.inputs, *leaves.values()],
                        [cotangent_curvatures[fn, n], *(d[fn, n] for d in cotangent_grads)],
                        retain_graph=True, materialize_grads=True,  # (the graph of the gradients is shared)
                    )
                    for fn in range(len(d_curvatures))
                ]
                d_pair = [torch.stack(d) for d in zip(*d_pair)]
                d_graph = d_pair if d_graph is None else [d + d_p for d, d_p in zip(d_graph, d_pair)]
                del curvature, pair_grads, d_pair
            d_graphs.append(d_graph)
        for d_graph in filter(None, d_graphs):
            d_inputs = [d + d_g.to(d.device) for d, d_g in zip(d_inputs, d_graph)]
            d_critical = [d + d_g.to(d.device) for d, d_g in zip(d_critical, d_graph[len(inputs) :])]
        del d_graphs

        shapes = {k: p.shape for k, p in self.params.items()}
        U = flatten(critical)  # [#params, k]
        off_span = lambda V: V - U @ (U.T @ V)
        minus_H = lambda V: -off_span(flatten(hvp_fn(unflatten(off_span(V), shapes))))
        Y = off_span(flatten({k: d.flatten(0, 1) for k, d in zip(shapes, d_critical)}))  # [#params, #fns * k]
        Z = minres(minus_H, Y, shift=eigvals[:n_critical].repeat(len(Y.T) // n_critical), rtol=self.VJP_RTOL)[0]
        H_critical = hvp_fn(critical, create_graph=True)
        d_eigvecs = [
            torch.autograd.grad(list(H_critical.values()), inputs, list(unflatten(Z_fn, shapes).values()), retain_graph=True, materialize_grads=True)
            for Z_fn in off_span(Z).split(n_critical, 1)
        ]
        return [d + torch.stack(d_eig) for d, d_eig in zip(d_inputs, zip(*d_eigvecs))]

    def precond(self, t: int) -> Mapping[str, Tensor | float]:
        return {k: self.lr(t) / self.SUBSTEPS for k in self.params}

    @property
    def stability_threshold(self) -> float:
        """2 / lr on the scale of the step lr / SUBSTEPS"""
        return 2 / self.SUBSTEPS


@dataclass(kw_only=True)
class NewtonState(SecondOrderState):
    """Damped Newton: `params <- params - lr * (H + shift I)^-1 (grads + wd * params)`.
    We set `shift = max(-2 min_eig, 0) + damping` so that the IVHP is well-conditioned.
    The backward pass (vjp) is custom and equivalent in cost to the forward pass (step)"""

    aux = ("newton", "eigvec", "stats")
    newton: dict[str, Tensor] = field(default_factory=dict)  # last IHVP solution (warm start)
    eigvec: dict[str, Tensor] = field(default_factory=dict)  # most negative eig (warm start)

    damping: float = 0.1
    rtol: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if not self.newton:
            self.newton = {k: torch.zeros_like(p) for k, p in self.params.items()}
        if not self.eigvec:
            g = torch.Generator().manual_seed(0)
            self.eigvec = {
                k: torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype)
                for k, p in self.params.items()
            }
        if not self.stats:
            p = next(iter(self.params.values()))
            nan = torch.tensor(float('nan'), dtype=p.dtype, device=p.device)
            self.stats = {
                "newton_iters": nan, "residual": nan, "min_eig": nan, "eig_iters": nan
            }

    def _solve(
        self, hvp_fn: HVP, shift: Tensor, rhs: dict[str, Tensor], x0: dict[str, Tensor] | None = None
    ) -> tuple[dict[str, Tensor], Tensor, int]:
        """Solve `(H + shift I)^-1 rhs` by MINRES with warm start `x0`"""
        shapes = {k: p.shape for k, p in self.params.items()}
        batched = next(iter(rhs.values())).ndim > len(next(iter(shapes.values())))
        H = lambda V: flatten(hvp_fn(unflatten(V, shapes)))  # H on flat columns [#params, m]
        X, residual, n_iters = minres(
            H, flatten(rhs, batched).detach(), shift=shift.detach(),
            x0=None if x0 is None else flatten(x0, batched), rtol=self.rtol
        )
        return unflatten(X, shapes), residual, n_iters

    def _shift(self, min_eig: Tensor) -> Tensor:
        return (-2 * min_eig).clamp_min(0) + self.damping

    def _update(self, t: int, grads: dict[str, Tensor], create_graph: bool = False) -> Self:
        hvp_fn = hvp_from_grads(grads, self.params)
        min_eig, eigvec, eig_iters = hessian_eigs(hvp_fn, self.eigvec, sign=-1, batched=False)
        shift = self._shift(min_eig)

        reg_grads = {k: g + self.wd * p for k, (p, g) in dict_zip(self.params, grads).items()}
        newton, res, n_iters = self._solve(hvp_fn, shift, reg_grads, x0=self.newton)
        params = {k: p - self.lr(t) * x for k, (p, x) in dict_zip(self.params, newton).items()}

        stats = {
            "newton_iters": torch.full_like(res, n_iters), "residual": res,
            "min_eig": min_eig, "eig_iters": torch.full_like(min_eig, eig_iters),
        }
        return replace(self, params=params, newton=newton, eigvec=eigvec, stats=stats)

    def vjp(
        self, t: int, forward_fn: Forward, inputs: Sequence[Tensor], cotangents: OptimState
    ) -> Sequence[Tensor]:
        """Implicit differentiation of a Newton step via one more IVHP.
        With `F = grads - (H + shift I) @ newton` and `adjoint = (H + shift I)^-1 @ cotangent`,
            `cotangent^T d(params - lr * newton) = cotangent^T d(params) - lr * adjoint^T dF`
        """
        grads = torch.autograd.grad(forward_fn(self.params)[0], self.params, create_graph=True)
        next_state = self._update(t, grads)
        newton, eigvec = next_state.newton, next_state.eigvec
        hvp_fn = hvp_from_grads(grads, self.params)
        reg_grads = {k: g + self.wd * p for k, (p, g) in dict_zip(self.params, grads).items()}

        H_eigvec, H_newton = hvp_fn(eigvec, create_graph=True), hvp_fn(newton, create_graph=True)
        min_eig = torch.stack([(v * Hv).sum() for v, Hv in dict_zip(eigvec, H_eigvec).values()]).sum()
        shift = self._shift(min_eig)  # with its graph
        F = {k: r - Hx - shift * x for k, (r, Hx, x) in dict_zip(reg_grads, H_newton, newton).items()}

        adjoint, _, _ = self._solve(hvp_fn, shift, cotangents.params)
        d_F = torch.autograd.grad(
            list(F.values()), inputs, list(adjoint.values()), is_grads_batched=True
        )
        d_params = torch.autograd.grad(
            list(self), inputs, list(cotangents), is_grads_batched=True, materialize_grads=True
        )
        return [dp - self.lr(t) * df for (dp, df) in zip(d_params, d_F)]

    def precond(self, t: int) -> Mapping[str, Tensor | float]:
        """The step size in low curvature directions (where Newton ~ SGD)"""
        shift = self._shift(self.stats["min_eig"])
        return {k: self.lr(t) / shift for k in self.params}
