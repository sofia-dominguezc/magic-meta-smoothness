"""Optimizer states: the model parameters + the optimizer's tensors, and the step on them (`OptimState.step`, a pure function
of the state and the gradients, so that MAGIC can differentiate through it), with the learning rate schedule."""

import math
from dataclasses import dataclass, field, replace
from typing import Callable, ClassVar, Iterator, Iterable, Mapping, Self, Sequence

import torch
from torch import Tensor

from src.minres import minres
from src.utils import HVP, dict_zip, hvp_from_grads
from src.lobpcg import flatten, hessian_eigs, unflatten


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
    aux: ClassVar[tuple[str, ...]] = ("ema_params",)

    params: dict[str, Tensor]  # model parameters
    lr: LRSchedule
    wd: float = 0.0  # weight decay
    ema_decay: float = 0.9
    ema_params: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.lr, dict):  # saved by `asdict`
            self.lr = LRSchedule(**self.lr)
        if not self.ema_params:
            self.ema_params = {k: p.detach().clone() for k, p in self.params.items()}

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

    def forward(
        self, forward_fn: Callable[[dict[str, Tensor], Tensor, Tensor], tuple[Tensor, Tensor]], X: Tensor, Y: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """The training forward: (per-example losses, outputs, aux_loss). The gradients are taken w.r.t. `params`;
        `aux_loss` (per-example, None by default) is weighted like the losses and given to `step` for its own use"""
        return *forward_fn(self.params, X, Y), None

    def _step(self, t: int, grads: dict[str, Tensor], aux_loss: Tensor | None = None) -> Self:
        """The state after step `t` with the gradients `grads`, without the EMA update"""
        raise NotImplementedError

    def _update_ema(self) -> Self:
        """The EMA moved towards the (new) params"""
        ema_params = {
            k: ema + (1 - self.ema_decay) * (p.detach() - ema)
            for k, (ema, p) in dict_zip(self.ema_params, self.params).items()
        }
        return replace(self, ema_params=ema_params)

    def step(self, t: int, grads: dict[str, Tensor], aux_loss: Tensor | None = None) -> Self:
        """The state after step `t` with the gradients `grads`"""
        return self._step(t, grads, aux_loss)._update_ema()

    def vjp(
        self, t: int, grads: dict[str, Tensor], inputs: Sequence[Tensor], cotangents: "OptimState", aux_loss: Tensor | None = None
    ) -> Sequence[Tensor]:
        """Backward of `step`: d(next_state)/d(inputs) at `contangents` (batched)"""
        next_state = self.step(t, grads, aux_loss)
        return torch.autograd.grad(
            list(next_state), inputs, list(cotangents), is_grads_batched=True
        )

    def gn_vjp(
        self, t: int, grads: dict[str, Tensor], inputs: Sequence[Tensor], cotangents: "OptimState", gnvp_fn: HVP
    ) -> tuple[Self, Sequence[Tensor]]:
        """Backward of `step` with the Hessian d(grads)/d(params) replaced by the Gauss-Newton matrix.
        Splits graph into `params, inputs -> grads` and `state, grads -> next_state`,
            but so that there's no graph `params -> grads` and instead we use `gnvp_fn`"""
        # backward of optim step
        grad_leaves = {k: g.detach().requires_grad_() for k, g in grads.items()}
        next_state = self.step(t, grad_leaves)
        d_all = torch.autograd.grad(
            list(next_state), [*self, *grad_leaves.values()], list(cotangents), is_grads_batched=True
        )
        n_state = len(d_all) - len(grads)
        d_state, d_grads = self.unflatten(d_all[:n_state]), dict(zip(grads, d_all[n_state:]))

        # backward of `params, inputs -> grads`
        d_params = {k: dp + Gc for k, (dp, Gc) in dict_zip(d_state.params, gnvp_fn(d_grads)).items()}
        d_inputs = torch.autograd.grad(
            list(grads.values()), inputs, list(d_grads.values()), is_grads_batched=True
        ) if inputs else ()
        return replace(d_state, params=d_params), d_inputs

    def precond(self, t: int) -> Mapping[str, Tensor | float]:
        """Per-element step size P. The update is P * grad"""
        return {k: self.lr(t) for k in self.params}

    @property
    def stability_threshold(self) -> float:
        """The EoS condition is `max_eig(sqrt(P) H sqrt(P)) = stability_threshold`"""
        raise NotImplementedError


@dataclass(kw_only=True)
class SGDState(OptimState):
    def _step(self, t: int, grads: dict[str, Tensor], aux_loss: Tensor | None = None) -> Self:
        params = {
            k: p - self.lr(t) * (g + self.wd * p)
            for k, (p, g) in dict_zip(self.params, grads).items()
        }
        return replace(self, params=params)

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

    def _step(self, t: int, grads: dict[str, Tensor], aux_loss: Tensor | None = None) -> Self:
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

    def _step(self, t: int, grads: dict[str, Tensor], aux_loss: Tensor | None = None) -> Self:
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
class SecondOrderState(OptimState):
    aux = ("ema_params", "stats")
    stats: dict[str, Tensor] = field(default_factory=dict)

    def _step(self, t: int, grads: dict[str, Tensor], aux_loss: Tensor | None = None, create_graph: bool = False) -> Self:
        """`create_graph`: of the HVPs, so that `vjp` can differentiate through them"""
        raise NotImplementedError

    def vjp(
        self, t: int, grads: dict[str, Tensor], inputs: Sequence[Tensor], cotangents: OptimState, aux_loss: Tensor | None = None
    ) -> Sequence[Tensor]:
        next_state = self._step(t, grads, aux_loss, create_graph=True)._update_ema()
        return torch.autograd.grad(
            list(next_state), inputs, list(cotangents), is_grads_batched=True
        )

    def gn_vjp(
        self, t: int, grads: dict[str, Tensor], inputs: Sequence[Tensor], cotangents: "OptimState", gnvp_fn: HVP
    ) -> tuple[Self, Sequence[Tensor]]:
        """`step` differentiates `grads` itself (HVPs), so they can't be swapped for leaves"""
        raise NotImplementedError(f"GN vjp isn't implemented for {type(self).__name__}")

    @property
    def stability_threshold(self) -> float:
        return 2.0


@dataclass(kw_only=True)
class SharpnessState(SecondOrderState):
    """SGD with step size `lr * 2 / sharpness`.
    Sharpness is computed with its graph, allowing backprop through a step"""

    aux = ("ema_params", "eigvec", "stats")
    eigvec: dict[str, Tensor] = field(default_factory=dict)  # top eigenvector (warm start)

    def __post_init__(self):
        super().__post_init__()
        if not self.eigvec:
            g = torch.Generator().manual_seed(0)
            self.eigvec = {
                k: torch.randn(p.shape, generator=g, dtype=p.dtype).to(p.device)
                for k, p in self.params.items()
            }
        if not self.stats:
            p = next(iter(self.params.values()))
            nan = torch.tensor(float("nan"), dtype=p.dtype, device=p.device)
            self.stats = {"sharpness": nan, "eig_iters": torch.zeros_like(nan)}

    def _step(self, t: int, grads: dict[str, Tensor], aux_loss: Tensor | None = None, create_graph: bool = False) -> Self:
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
    """Edge Flow (Marion 2026, Algorithm 1): `lr` is the modeled GD learning rate eta; each GD step is SUBSTEPS steps of
    size rho = eta / SUBSTEPS of the center `params` (gradient averaged over `params` +- x eigvec), the oscillation
    direction `eigvec` (a power iteration, unit) and magnitude `x` (grows iff the sharpness |H eigvec| > 2 / eta)"""

    SUBSTEPS: ClassVar[int] = 4
    X_MIN: ClassVar[float] = 1e-4  # floor of x

    groups = ("params", "eigvec", "x")
    eigvec: dict[str, Tensor] = field(default_factory=dict)  # None -> random unit
    x: dict[str, Tensor] = field(default_factory=dict)  # {"x": []}; None -> X_MIN

    def __post_init__(self):
        super().__post_init__()
        p = next(iter(self.params.values()))
        if not self.eigvec:
            g = torch.Generator().manual_seed(0)
            u = {k: torch.randn(p.shape, generator=g, dtype=p.dtype).to(p.device) for k, p in self.params.items()}
            norm = torch.sqrt(sum((v * v).sum() for v in u.values()))
            self.eigvec = {k: v / norm for k, v in u.items()}
        if not self.x:
            self.x = {"x": torch.full((), self.X_MIN, dtype=p.dtype, device=p.device)}
        if not self.stats:
            nan = torch.tensor(float("nan"), dtype=p.dtype, device=p.device)
            self.stats = {"sharpness": nan, "x": self.x["x"].detach(), "center_loss": nan}

    def forward(self, forward_fn, X: Tensor, Y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Losses averaged over the endpoints `params` +- x eigvec; outputs and aux_loss at the center `params`"""
        x = self.x["x"]
        ends = [{k: p + s * x * v for k, (p, v) in dict_zip(self.params, self.eigvec).items()} for s in (1, -1)]
        (l_plus, _), (l_minus, _) = (forward_fn(p, X, Y) for p in ends)
        l_center, out = forward_fn(self.params, X, Y)
        return (l_plus + l_minus) / 2, out, l_center

    def _step(self, t: int, grads: dict[str, Tensor], aux_loss: Tensor | None = None, create_graph: bool = False) -> Self:
        assert aux_loss is not None, "EdgeFlowState.step needs the center loss from its forward"
        center_grads = torch.autograd.grad(aux_loss, self.params, create_graph=True)
        H_eigvec = hvp_from_grads(center_grads, self.params)(self.eigvec, create_graph=create_graph)
        sharpness = torch.cat([(v * hv).sum() for (v, hv) in dict_zip(self.eigvec, H_eigvec).values()]).sum()
        rho = self.lr(t) / self.SUBSTEPS  # Edge-Flow step size
        params = {k: p - rho * (g + self.wd * p) for k, (p, g) in dict_zip(self.params, grads).items()}
        eigvec = {k: hv / sharpness for k, hv in H_eigvec.items()}
        x = {"x": (self.x["x"] * (1 + rho * (sharpness - 2 / self.lr(t)))).clamp_min(self.X_MIN)}
        stats = {"sharpness": sharpness.detach(), "x": x["x"].detach(), "center_loss": aux_loss.detach()}
        return replace(self, params=params, eigvec=eigvec, x=x, stats=stats)

    def precond(self, t: int) -> Mapping[str, Tensor | float]:
        return {k: self.lr(t) / self.SUBSTEPS for k in self.params}

    @property
    def stability_threshold(self) -> float:
        """2 / eta on the scale of the step rho"""
        return 2 / self.SUBSTEPS


@dataclass(kw_only=True)
class NewtonState(SecondOrderState):
    """Damped Newton: `params <- params - lr * (H + shift I)^-1 (grads + wd * params)`.
    We set `shift = max(-2 min_eig, 0) + damping` so that the IVHP is well-conditioned.
    The backward pass (vjp) is custom and equivalent in cost to the forward pass (step)"""

    aux = ("ema_params", "newton", "eigvec", "stats")
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
                k: torch.randn(p.shape, generator=g, dtype=p.dtype).to(p.device)
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
        return (-3 * min_eig).clamp_min(0) + self.damping

    def _step(self, t: int, grads: dict[str, Tensor], aux_loss: Tensor | None = None, create_graph: bool = False) -> Self:
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
        self, t: int, grads: dict[str, Tensor], inputs: Sequence[Tensor], cotangents: OptimState, aux_loss: Tensor | None = None
    ) -> Sequence[Tensor]:
        """Implicit differentiation of a Newton step via one more IVHP.
        With `F = grads - (H + shift I) @ newton` and `adjoint = (H + shift I)^-1 @ cotangent`,
            `cotangent^T d(params - lr * newton) = cotangent^T d(params) - lr * adjoint^T dF`
        """
        next_state = self.step(t, grads)
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
