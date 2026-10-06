from dataclasses import dataclass, replace, fields
import math
from typing import Callable, Iterator, Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call
from tqdm import tqdm

from src.data import Dataset
from src.lobpcg import torch_lobpcg
from src.utils import Logger

LossFn = Callable[[Tensor, dict[str, Tensor]], Tensor]  # (model output, batch) -> per-example losses [B]


def mse(out: Tensor, batch: dict[str, Tensor]) -> Tensor:
    assert out.shape == batch["y"].shape
    return (out - batch["y"]).pow(2).flatten(1).mean(1)


def cross_entropy(out: Tensor, batch: dict[str, Tensor]) -> Tensor:
    return F.cross_entropy(out, batch["y"], reduction="none")


LOSSES: dict[str, LossFn] = {"mse": mse, "ce": cross_entropy}


def batch_loss(
    model: nn.Module, loss_fn: LossFn, params: dict[str, Tensor], batch: dict[str, Tensor], w: Tensor
) -> tuple[Tensor, Tensor]:
    """Weighted loss sum(w_i * loss_i) / B, and the model output."""
    out = functional_call(model, params, (batch["x"],))
    return (w * loss_fn(out, batch)).mean(), out


def subset_loss(
    model: nn.Module, loss_fn: LossFn, params: dict[str, Tensor], data: Dataset, idx: Tensor, chunk: int = 1024
) -> tuple[Tensor, dict[str, Tensor]]:
    """Mean loss over data[idx] and its gradient wrt `params`.
    Computed in chunks of `chunk` examples to bound the memory."""
    params = {k: p.detach().requires_grad_() for k, p in params.items()}
    value, grads = torch.zeros((), device=data.device), {k: torch.zeros_like(p) for k, p in params.items()}
    for i in range(0, len(idx), chunk):
        batch = data[idx[i : i + chunk]]
        term = loss_fn(functional_call(model, params, (batch["x"],)), batch).sum() / len(idx)
        dgrads = torch.autograd.grad(term, params)
        grads = {k: grads[k] + dgrads[k] for k in grads}
        value += term.detach()
    return value, grads


@dataclass
class Loader:
    """Random batches (with replacement), deterministic in `step`, with per-example weights."""

    dataset: Dataset
    batch_size: int = 32
    seed: int = 0
    weights: Tensor | None = None  # [N]; None -> all ones

    def __post_init__(self):
        if self.weights is None:
            self.weights = torch.ones(len(self.dataset))
        self.weights = self.weights.to(self.dataset.device)

    def get(self, step: int) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        """Returns (batch, indices, weights) for `step`."""
        assert self.weights is not None
        g = torch.Generator().manual_seed((self.seed << 32) + step)
        idx = torch.randint(len(self.dataset), (self.batch_size,), generator=g).to(self.dataset.device, non_blocking=True)
        return self.dataset[idx], idx, self.weights[idx]


def dict_zip[T](*dicts: dict[str, T]) -> dict[str, tuple[T, ...]]:
    """e.g. {"a": 1, "b": 2}, {"a": 3, "b": 4} -> {"a": (1, 3), "b": (2, 4)}"""
    assert len(dicts) > 0
    zipped_dict: dict[str, tuple[T, ...]] = {}
    for key in dicts[0]:
        zipped_dict[key] = tuple(d[key] for d in dicts)
    return zipped_dict


@dataclass(kw_only=True)
class OptimState:
    """General class for model + optimizer states"""

    params: dict[str, Tensor]  # model parameters

    @property
    def _state_tensors(self) -> dict[str, dict[str, Tensor]]:
        """All state tensors. Updated in `step`, and are differentiable."""
        tensor_fields: dict[str, dict[str, Tensor]] = {}
        for field in fields(self):
            if isinstance(tensor_dict := getattr(self, field.name), dict):
                tensor_fields[field.name] = tensor_dict
        return tensor_fields

    @property
    def all_tensors(self) -> list[Tensor]:
        all_tensors: list[Tensor] = []
        for _, tensor_dict in self._state_tensors.items():
            all_tensors.extend(tensor_dict.values())
        return all_tensors

    def apply(self, fn: Callable[[Tensor], Tensor]) -> Self:
        new_fields = {
            field_name: {name: fn(tensor) for name, tensor in tensor_dict.items()}
            for field_name, tensor_dict in self._state_tensors.items()
        }
        return replace(self, **new_fields)

    def step(self, grads: dict[str, Tensor]) -> Self:
        raise NotImplementedError


@dataclass(kw_only=True)
class SGDState(OptimState):
    lr: float
    weight_decay: float = 0.0

    def step(self, grads: dict[str, Tensor]) -> Self:
        new_params = {
            name: p - self.lr * (g + self.weight_decay * p)
            for name, (p, g) in dict_zip(self.params, grads).items()
        }
        return replace(self, params=new_params)


@dataclass(kw_only=True)
class SGDMomentumState(SGDState):
    momentum: float = 0.9
    buffer: dict[str, Tensor] | None = None  # None -> zeros

    def __post_init__(self):
        if self.buffer is None:
            self.buffer = {k: torch.zeros_like(p) for k, p in self.params.items()}

    def step(self, grads: dict[str, Tensor]) -> Self:  # same as torch.optim.SGD (dampening=0, nesterov=False)
        assert self.buffer is not None
        new_buffer = {
            name: self.momentum * ewma_p + g + self.weight_decay * p
            for name, (p, ewma_p, g) in dict_zip(self.params, self.buffer, grads).items()
        }
        new_params = {
            name: p - self.lr * ewma_p
            for name, (p, ewma_p) in dict_zip(self.params, new_buffer).items()
        }
        return replace(self, params=new_params, buffer=new_buffer)


# ----- training -------------------------------------------------


def train(
    model: nn.Module,
    loader: Loader,
    loss_fn: LossFn,
    state: OptimState,
    *,
    start: int = 0,
    n_steps: int,
    ckpt_every: int | None = None,
    logger: Logger | None = None,
    num_top_eigs: int = 0,
    progress: bool = True,
) -> tuple[OptimState, list[OptimState]]:
    """Train deterministically on steps [start, start + n_steps).
    Returns the final state and the states before steps start, start + ckpt_every, ...
    """
    state = state.apply(lambda t: t.to(loader.dataset.device, non_blocking=True))
    ckpts: list[OptimState] = []
    eigvecs: Tensor | None = None
    for t in tqdm(range(start, start + n_steps), disable=not progress):
        batch, _, w = loader.get(t)
        params = {k: p.detach().requires_grad_() for k, p in state.params.items()}
        loss, out = batch_loss(model, loss_fn, params, batch, w)
        grads = torch.autograd.grad(loss, params, create_graph=num_top_eigs > 0)

        if logger is not None:
            metrics = loader.dataset.metrics(out.detach(), batch)
            logger.log(t, loss=loss.detach(), **{k: v.mean() for k, v in metrics.items()})

        if ckpt_every and (t - start) % ckpt_every == 0:
            ckpts.append(state.apply(lambda t: t.to("cpu")))

        if num_top_eigs > 0:  # top Hessian eigenvalues via LOBPCG on the HVP operator
            sizes = [p.numel() for p in params.values()]

            def hvp(V: Tensor) -> Tensor:  # [P, r] -> [P, r]
                cols = []
                for v in V.unbind(1):
                    vs = [c.view_as(p) for c, p in zip(v.split(sizes), params.values())]
                    hv = torch.autograd.grad(list(grads.values()), params, grad_outputs=vs, retain_graph=True)
                    cols.append(torch.cat([h.flatten() for h in hv.values()]))
                return torch.stack(cols, 1)

            if eigvecs is None:
                g = torch.Generator().manual_seed(start)
                eigvecs = torch.randn(sum(sizes), num_top_eigs, generator=g).to(loader.dataset.device, non_blocking=True)
            eigvals, eigvecs = torch_lobpcg(hvp, eigvecs)
            if logger is not None:
                logger.log(t, eigvals=eigvals)
            grads = {name: g.detach() for name, g in grads.items()}

        state = state.step(grads)
    return state, ckpts


def reversed_states(
    model: nn.Module, loader: Loader, loss_fn: LossFn,
    init_state: OptimState, *, n_steps: int,
) -> Iterator[OptimState]:
    """Reconstruct the model states in reverse order by running training
    twice, storing `sqrt(n_steps)` checkpoints on each pass.
    """
    sqrt_n_steps = math.ceil(math.sqrt(n_steps))
    final, ckpts = train(
        model, loader, loss_fn, init_state, n_steps=n_steps,
        ckpt_every=sqrt_n_steps, progress=False,
    )
    yield final
    for i, ckpt in reversed(list(enumerate(ckpts))):
        start = i * sqrt_n_steps
        _, inner = train(
            model, loader, loss_fn, ckpt, start=start,
            n_steps=min(sqrt_n_steps, n_steps - start),
            ckpt_every=1, progress=False,
        )
        yield from reversed(inner)


def magic(
    model: nn.Module,
    loader: Loader,
    loss_fn: LossFn,
    init_state: OptimState,
    n_steps: int,
    test_dataset: Dataset,
    eval_idxs: list[Tensor],
) -> tuple[OptimState, list[Tensor], list[Tensor]]:
    """Run MAGIC for the eval functions "mean test loss over test_dataset[idx]".
    Computational cost: `3 + #fns` trainings.
    Max memory usage: `sqrt(T) + #fns` model states.

    Returns (s_T, list of f(s_T), list of df(s_T)/dw)
    """
    all_states = reversed_states(  # 2x training
        model, loader, loss_fn, init_state, n_steps=n_steps
    )
    final = next(all_states)
    if len(eval_idxs) == 0:
        return final, [], []

    zero_state = final.apply(lambda t: torch.zeros_like(t))
    outputs: list[Tensor] = []       # f(s_t)
    deltas: list[list[Tensor]] = []  # df/ds_t
    betas: list[Tensor] = []         # df/dw
    for idx in eval_idxs:
        f_value, f_grad = subset_loss(model, loss_fn, final.params, test_dataset, idx)
        outputs.append(f_value)
        deltas.append(replace(zero_state, params=f_grad).all_tensors)
        betas.append(torch.zeros(len(loader.dataset), device=loader.dataset.device))

    for t, state in zip(reversed(range(n_steps)), tqdm(all_states, total=n_steps)):
        batch, indices, w = loader.get(t)
        w = w.requires_grad_()
        state = state.apply(lambda t: t.to(w.device, non_blocking=True).detach().requires_grad_())
        loss, _ = batch_loss(model, loss_fn, state.params, batch, w)
        grads = torch.autograd.grad(loss, state.params, create_graph=True)  # 1x training
        new_state = state.step(grads)
        for i, delta in enumerate(deltas):
            # (df/ds_t, df/dw_t) = ds_{t+1} / d(s_t, w_t) @ df/ds_{t+1}
            *d_state, d_w = torch.autograd.grad(  # 1x training per `delta`
                new_state.all_tensors,
                [*state.all_tensors, w],
                delta,
                retain_graph=i < len(deltas) - 1,
            )
            deltas[i] = d_state
            betas[i].index_add_(0, indices, d_w)

    return final, outputs, betas
