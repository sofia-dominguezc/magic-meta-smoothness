"""The model's losses and metrics, and their derivatives: `eval_loss` (over many examples) and `hvp_fn` (Hessian-vector products)."""

from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.func import grad_and_value, functional_call

from src.data import Loader
from src.eigvals import HVP


LossFn = Callable[[Tensor, Tensor], Tensor]  # (model output [B, C], int labels [B]) -> per-example losses [B]


def cross_entropy(out: Tensor, Y: Tensor) -> Tensor:
    return F.cross_entropy(out, Y, reduction="none")


def mse(out: Tensor, Y: Tensor) -> Tensor:
    """Squared error to the one-hot labels"""
    return 0.5 * (out - F.one_hot(Y, out.shape[1]).to(out.dtype)).pow(2).sum(-1)


LOSSES: dict[str, LossFn] = {"ce": cross_entropy, "mse": mse}


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


def eval_loss(
    model: nn.Module, loss_fn: LossFn, params: dict[str, Tensor], loader: Loader, indices: Tensor,
    *, grad: bool = False, bf16: bool = False,
) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    """Mean loss of `params` over the examples `indices` of `loader`, accumulated batch by batch to bound the memory.
    Returns: (loss, {metric: mean value}, {param: d loss / d param} if `grad` else {})."""
    n = len(indices)

    def batch_loss(params: dict[str, Tensor], X: Tensor, Y: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """This batch's share of the mean loss, and of the mean metrics."""
        out = model_call(model, params, X, bf16=bf16)
        return loss_fn(out, Y).sum() / n, {k: v.sum() / n for k, v in metrics(out, Y).items()}

    params = {k: p.detach() for k, p in params.items()}
    loss, mean_metrics = torch.zeros((), device=loader.device), {}
    grads = {k: torch.zeros_like(p) for k, p in params.items()} if grad else {}
    for X, Y in loader.iter_batches(indices):
        if grad:
            d_params, (term, batch_metrics) = grad_and_value(batch_loss, has_aux=True)(params, X, Y)
            grads = {k: grads[k] + d_params[k] for k in grads}
        else:
            term, batch_metrics = batch_loss(params, X, Y)
        loss += term.detach()
        for k, v in batch_metrics.items():
            mean_metrics[k] = mean_metrics.get(k, 0) + v.detach()
    return loss, mean_metrics, grads


def hvp_fn(
    model: nn.Module, loss_fn: LossFn, params: dict[str, Tensor], X: Tensor, Y: Tensor,
    *, grads: dict[str, Tensor] | None = None, bf16: bool = False,
) -> HVP:
    """Build the V -> HV function (V with a leading batch dim) of the mean loss on (X, Y) using stateful 2nd-order autograd,
    from `grads` if they were already computed with `create_graph` at the leaves `params` (as yielded by `Magic.train.trace`)."""
    if grads is None:
        params = {k: p.detach().requires_grad_() for k, p in params.items()}
        out = model_call(model, params, X, bf16=bf16)
        grads = torch.autograd.grad(loss_fn(out, Y).mean(), params, create_graph=True)

    def hvp(V: dict[str, Tensor]) -> dict[str, Tensor]:
        return torch.autograd.grad(
            list(grads.values()), params, [V[k] for k in grads],
            retain_graph=True, is_grads_batched=True,
        )

    return hvp
