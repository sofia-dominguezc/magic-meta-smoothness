"""The model call, losses and metrics, and Hessian-vector products from gradients."""

from typing import Callable, Mapping, Protocol

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.func import functional_call


LossFn = Callable[[Tensor, Tensor], Tensor]  # (model output [B, C], int labels [B]) -> per-example losses [B]


class HVP(Protocol):
    """V {name: [m, *shape]} -> H V (batched over the leading dim m), or a single vector {name: [*shape]} -> H v;
    `create_graph=True` for a differentiable H V"""
    def __call__(self, V: dict[str, Tensor], create_graph: bool = False) -> dict[str, Tensor]: ...


def hvp_from_grads(grads: dict[str, Tensor], params: dict[str, Tensor]) -> HVP:
    """V -> H V = d(grads . V) / d params by stateful 2nd-order autograd: `grads` computed with create_graph at the leaves `params`"""
    def hvp_fn(V: dict[str, Tensor], create_graph: bool = False) -> dict[str, Tensor]:
        batched = next(iter(V.values())).ndim > next(iter(params.values())).ndim
        return torch.autograd.grad(
            list(grads.values()), params, [V[k] for k in grads],
            retain_graph=True, create_graph=create_graph, is_grads_batched=batched,
        )
    return hvp_fn


def dict_zip[T](*dicts: Mapping[str, T]) -> dict[str, tuple[T, ...]]:
    """e.g. {"a": 1, "b": 2}, {"a": 3, "b": 4} -> {"a": (1, 3), "b": (2, 4)}"""
    assert all(set(dicts[0]) == set(d) for d in dicts)
    return {key: tuple(d[key] for d in dicts) for key in dicts[0]}


def cross_entropy(out: Tensor, Y: Tensor) -> Tensor:
    return F.cross_entropy(out, Y, reduction="none")


def mse(out: Tensor, Y: Tensor) -> Tensor:
    """Squared error to the one-hot labels (built by comparison, as `F.one_hot` doesn't support vmap)"""
    one_hot = (Y[:, None] == torch.arange(out.shape[1], device=out.device)).to(out.dtype)
    return 0.5 * (out - one_hot).pow(2).sum(-1)


LOSSES: dict[str, LossFn] = {"ce": cross_entropy, "mse": mse}


def cross_entropy_output_hessian(out: Tensor, Y: Tensor, U: Tensor) -> Tensor:
    """H_L U per example for the cross-entropy: (diag(p) - p p^T) u with p = softmax(out)"""
    p = out.softmax(-1)
    return p * (U - (p * U).sum(-1, keepdim=True))


def mse_output_hessian(out: Tensor, Y: Tensor, U: Tensor) -> Tensor:
    return U


OUTPUT_HESSIANS: dict[LossFn, Callable[[Tensor, Tensor, Tensor], Tensor]] = {  # loss -> (out [B, C], Y [B], U [B, C]) -> H_L U [B, C]
    cross_entropy: cross_entropy_output_hessian, mse: mse_output_hessian,
}


def metrics(out: Tensor, Y: Tensor) -> dict[str, Tensor]:
    """Per-example classification metrics of the model: accuracy and margin."""
    correct = out.gather(1, Y[:, None])[:, 0]
    best_other = out.scatter(1, Y[:, None], float("-inf")).max(1).values
    return {"acc": (out.argmax(1) == Y).to(out.dtype), "margin": correct - best_other}


def model_call(model: nn.Module, params: dict[str, Tensor], X: Tensor, *, bf16: bool = False) -> Tensor:
    """`model(params, X)`, with the matmuls in bfloat16 (autocast) if `bf16`"""
    with torch.autocast(X.device.type, torch.bfloat16, enabled=bf16):
        out: Tensor = functional_call(model, params, (X,))
    return out.to(X.dtype)
