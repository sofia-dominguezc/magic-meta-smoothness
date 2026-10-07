from dataclasses import dataclass, replace, fields
import math
from typing import Callable, Iterator, Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call
from tqdm import tqdm

from src.data import EvalLoader, Loader
from src.lobpcg import torch_lobpcg
from src.utils import Logger, dict_zip

LossFn = Callable[[Tensor, Tensor], Tensor]  # (model output, Y) -> per-example losses [B]


def mse(out: Tensor, Y: Tensor) -> Tensor:
    assert out.shape == Y.shape
    return (out - Y).pow(2).flatten(1).mean(1)


def cross_entropy(out: Tensor, Y: Tensor) -> Tensor:
    return F.cross_entropy(out, Y, reduction="none")


LOSSES: dict[str, LossFn] = {"mse": mse, "ce": cross_entropy}


def eval_loss(
    model: nn.Module, loss_fn: LossFn, params: dict[str, Tensor], loader: EvalLoader, grad: bool = False
) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    """Mean loss and mean dataset metrics over the examples of `loader`, and (if `grad`,
    else {}) the gradient of the loss wrt `params`, accumulated batch by batch to bound the memory."""
    params = {k: p.detach().requires_grad_(grad) for k, p in params.items()}
    loss, metrics = torch.zeros((), device=loader.device), {}
    grads = {k: torch.zeros_like(p) for k, p in params.items()} if grad else {}
    for X, Y in loader:
        out = functional_call(model, params, (X,))
        term = loss_fn(out, Y).sum() / len(loader)
        if grad:
            dgrads = torch.autograd.grad(term, params)
            grads = {k: grads[k] + dgrads[k] for k in grads}
        loss += term.detach()
        for k, v in loader.dataset.metrics(out.detach(), Y).items():
            metrics[k] = metrics.get(k, 0) + v.sum() / len(loader)
    return loss, metrics, grads


@dataclass(kw_only=True)
class OptimState:
    """General class containing all model + optimizer states"""

    params: dict[str, Tensor]  # model parameters

    @property
    def state_tensors(self) -> dict[str, dict[str, Tensor]]:
        tensor_fields: dict[str, dict[str, Tensor]] = {}
        for field in fields(self):
            if isinstance(tensor_dict := getattr(self, field.name), dict):
                tensor_fields[field.name] = tensor_dict
        return tensor_fields

    @property
    def all_tensors(self) -> list[Tensor]:
        """Return all state tensors as a flattened list"""
        all_tensors: list[Tensor] = []
        for _, tensor_dict in self.state_tensors.items():
            all_tensors.extend(tensor_dict.values())
        return all_tensors

    def apply(self, fn: Callable[[Tensor], Tensor]) -> Self:
        """Apply `fn` to each state tensor. Returns a new OptimState"""
        new_fields = {
            field_name: {name: fn(tensor) for name, tensor in tensor_dict.items()}
            for field_name, tensor_dict in self.state_tensors.items()
        }
        return replace(self, **new_fields)

    def unflatten(self, tensors: list[Tensor]) -> Self:
        """Inverse of `all_tensors`: a state like this one holding `tensors` instead"""
        it = iter(tensors)
        return self.apply(lambda _: next(it))

    def step(self, grads: dict[str, Tensor]) -> Self:
        """Update state tensors. Returns a new OptimState"""
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
            name: self.momentum * buf + g + self.weight_decay * p
            for name, (p, buf, g) in dict_zip(self.params, self.buffer, grads).items()
        }
        new_params = {
            name: p - self.lr * buf
            for name, (p, buf) in dict_zip(self.params, new_buffer).items()
        }
        return replace(self, params=new_params, buffer=new_buffer)


# ----- training -------------------------------------------------


def train(
    model: nn.Module,
    loader: Loader,
    loss_fn: LossFn,
    init_state: OptimState,
    *,
    start: int = 0,
    n_steps: int,
    ckpt_every: int | None = None,
    logger: Logger | None = None,
    eval_loader: EvalLoader | None = None,
    eval_every: int = 100,
    num_top_eigs: int = 0,
    progress: bool = True,
) -> tuple[OptimState, list[OptimState]]:
    """Train deterministically on steps [start, start + n_steps).
    Returns the final state and the states before steps start, start + ckpt_every, ...

    Optionally logs metrics of interest, including the top Hessian eigenvalues.
    """
    if eval_loader is not None or num_top_eigs > 0:
        assert logger is not None

    def eval_log(step: int, state: OptimState):
        assert logger is not None and eval_loader is not None
        loss, metrics, _ = eval_loss(model, loss_fn, state.params, eval_loader)
        logger.log(step, **{"eval/loss": loss}, **{f"eval/{k}": v for k, v in metrics.items()})

    state = init_state.apply(lambda t: t.to(loader.device, non_blocking=True))
    ckpts: list[OptimState] = []
    eigvecs: Tensor | None = None
    for t in tqdm(range(start, start + n_steps), disable=not progress):
        X, Y, _, weights = loader.get(t)
        params = {k: p.detach().requires_grad_() for k, p in state.params.items()}

        logits: Tensor = functional_call(model, params, (X,))
        loss = (weights * loss_fn(logits, Y)).mean()
        grads = torch.autograd.grad(loss, params, create_graph=num_top_eigs > 0)

        if ckpt_every and (t - start) % ckpt_every == 0:
            ckpts.append(state.apply(lambda t: t.to("cpu", copy=True)))
        if logger is not None:
            metrics = loader.dataset.metrics(logits.detach(), Y)
            logger.log(
                t, loss=loss.detach(), **{k: v.mean() for k, v in metrics.items()}
            )
        if num_top_eigs > 0:
            assert logger is not None
            sizes = [p.numel() for p in params.values()]
            grads_list = list(grads.values())

            def hvp(V: Tensor) -> Tensor:  # (#params, #eig) -> (#params, #eig)
                cols = []
                for v in V.unbind(1):
                    vs = [c.view_as(p) for c, p in zip(v.split(sizes), params.values())]
                    hv = torch.autograd.grad(grads_list, params, vs, retain_graph=True)
                    cols.append(torch.cat([h.flatten() for h in hv.values()]))
                return torch.stack(cols, 1)

            if eigvecs is None:
                logger.labels["eig"] = [f"eig {i + 1}" for i in range(num_top_eigs)]
                g = torch.Generator().manual_seed(start)
                eigvecs = torch.randn(
                    sum(sizes), num_top_eigs, generator=g
                ).to(loader.device, non_blocking=True)
            eigvals, next_eigvecs, n_iter = torch_lobpcg(hvp, eigvecs)
            logger.log(t, **{"eig/values": eigvals, "eig/lobpcg_iters": torch.tensor(n_iter)})
            logger.log(t, **{"eig/cos_to_next": (eigvecs * next_eigvecs).sum(0).abs()})
            eigvecs = next_eigvecs
            grads = {name: g.detach() for name, g in grads.items()}

        if eval_loader is not None and t % eval_every == 0:
            eval_log(t, state)
        state = state.step(grads)

    if eval_loader is not None:
       eval_log(start + n_steps, state)
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
    eval_loaders: list[EvalLoader],
    logger: Logger | None = None,
    eval_names: list[str] | None = None,
    num_top_betas: int = 4,
) -> tuple[list[Tensor], list[Tensor]]:
    """Run MAGIC for the eval functions "mean loss over the examples of each eval loader".
    Computational cost: `3 + #fns` trainings.
    Max memory usage: `sqrt(T) + #fns` model states.

    If `logger` is given, logs at each step t (going backward from T):
    - "delta/<field>": the RMS of df/ds_t restricted to each tensor dict of the state (e.g.
      params, buffer), one entry per eval fn;
    - "betas/top <k> df/dw, <eval name>": the k = `num_top_betas` largest |df/dw| accumulated over
      the steps >= t, by rank, with their training example indices (same key + "_ids").

    Returns (list of f(s_T), list of df(s_T)/dw)
    """
    eval_names = eval_names or [f"eval fn {i + 1}" for i in range(len(eval_loaders))]
    if len(eval_loaders) == 0:
        return [], []
    all_states = reversed_states(model, loader, loss_fn, init_state, n_steps=n_steps)
    final_state = next(all_states)  # (in device)  # 1x training

    zero_state = final_state.apply(lambda t: torch.zeros_like(t))
    if logger is not None:
        logger.labels["betas"] = [f"top {j + 1}" for j in range(num_top_betas)]
    outputs: list[Tensor] = []       # f(s_t)
    deltas: list[list[Tensor]] = []  # df/ds_t (as a flattened OptimState)
    betas: list[Tensor] = []         # df/dw
    for eval_loader in eval_loaders:
        f_value, _, f_grad = eval_loss(
            model, loss_fn, final_state.params, eval_loader, grad=True
        )
        outputs.append(f_value)
        deltas.append(replace(zero_state, params=f_grad).all_tensors)
        betas.append(torch.zeros(len(loader.dataset), device=loader.device))

    t = n_steps - 1
    for state in tqdm(all_states, total=n_steps):  # 1x training
        X, Y, indices, weights = loader.get(t)
        weights = weights.requires_grad_()
        state = state.apply(
            lambda t: t.to(loader.device, non_blocking=True).detach().requires_grad_()
        )

        logits: Tensor = functional_call(model, state.params, (X,))  # 1x fwd
        loss = (weights * loss_fn(logits, Y)).mean()
        grads = torch.autograd.grad(loss, state.params, create_graph=True)  # 1x bwd
        new_state = state.step(grads)

        for i, delta in enumerate(deltas):
            # (df/ds_t, df/dw_t) = ds_{t+1} / d(s_t, w_t) @ df/ds_{t+1}
            *d_state, d_weights = torch.autograd.grad(  # 1x double-bwd per `delta`
                new_state.all_tensors,
                [*state.all_tensors, weights],
                delta,
                retain_graph=i < len(deltas) - 1,
            )
            deltas[i] = d_state
            betas[i].index_add_(0, indices, d_weights)

        if logger is not None:
            delta_states = [state.unflatten(delta) for delta in deltas]
            for name in state.state_tensors:
                rms = torch.stack([
                    (sum(d.pow(2).sum() for d in ds.state_tensors[name].values())
                     / sum(d.numel() for d in ds.state_tensors[name].values())) ** 0.5
                    for ds in delta_states
                ])
                logger.log(t, **{f"delta/{name}": rms})
            for eval_name, beta in zip(eval_names, betas):
                ids = beta.abs().topk(min(num_top_betas, len(beta))).indices
                key = f"betas/top {num_top_betas} df/dw, {eval_name}"
                logger.log(t, **{key: beta[ids], f"{key}_ids": ids})
        t -= 1

    return outputs, betas
