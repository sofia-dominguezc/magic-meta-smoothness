"""Optimizer states: the model parameters + the optimizer's tensors, and the step on them (`OptimState.step`, a pure function
of the state and the batch loss, so that MAGIC can differentiate through it), with the learning rate schedule."""

import math
from dataclasses import dataclass, field, replace
from typing import Callable, ClassVar, Iterator, Iterable, Mapping, Self, Sequence

import torch
from torch import Tensor

from src.utils import dict_zip

Forward = Callable[[dict[str, Tensor]], tuple[Tensor, Tensor]]  # params -> (batch loss [(R,)], outputs [(R,) B, C])
StepOutput = tuple[dict[str, Tensor], Tensor, Tensor]  # (grads, loss, outputs)


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

    def cotangent(self, d_params: dict[str, Tensor]) -> Self:
        """A (batched) OptimState with `d_params` in the `params` field and zero elsewhere"""
        n_fns = len(next(iter(d_params.values())))
        zero_state = self.apply(lambda t: t.new_zeros(n_fns, *t.shape))
        return replace(zero_state, params=d_params)

    def step(self, t: int, forward_fn: Forward, create_graph: bool = False) -> tuple[Self, *StepOutput]:
        """Step `t` on the loss of `forward_fn` (one loss per model if the state is batched).
        `create_graph`: make the step differentiable"""
        loss, out = forward_fn(self.params)  # () or (#models,)
        grads = torch.autograd.grad(loss.sum(), self.params, create_graph=create_graph)
        return self._update(t, grads), grads, loss, out

    def _update(self, t: int, grads: dict[str, Tensor]) -> Self:
        """The state after step `t` with the gradients `grads`"""
        raise NotImplementedError

    def vjp(self, t: int, forward_fn: Forward, inputs: Sequence[Tensor], cotangents: "OptimState", next_state: "OptimState | None" = None) -> Sequence[Tensor]:
        """Backward of `step`: d(next_state)/d(inputs) at `cotangents` (batched); `next_state` = what `step` returned (detached), if known"""
        stepped = self.step(t, forward_fn, create_graph=True)[0]
        return torch.autograd.grad(list(stepped), inputs, list(cotangents), is_grads_batched=True)

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
