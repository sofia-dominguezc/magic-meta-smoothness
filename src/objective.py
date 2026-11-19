"""The loss as a function of the model parameters on given examples (`Objective`), with its derivatives, and the losses and metrics."""

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.func import functional_call

from src.data import Loader
from src.utils import HVP, hvp_from_grads


class LossFn(Protocol):
    """out: [B, C], int labels: [B] -> per-example loss: [B]"""

    def __call__(self, out: Tensor, Y: Tensor) -> Tensor: ...


def cross_entropy(out: Tensor, Y: Tensor) -> Tensor:
    return F.cross_entropy(out, Y, reduction="none")


def mse(out: Tensor, Y: Tensor) -> Tensor:
    """Squared error to the one-hot labels (built by comparison, as `F.one_hot` doesn't support vmap)"""
    one_hot = (Y[:, None] == torch.arange(out.shape[1], device=out.device)).to(out.dtype)
    return 0.5 * (out - one_hot).pow(2).sum(-1)


LOSSES: dict[str, LossFn] = {"ce": cross_entropy, "mse": mse}


def metrics(out: Tensor, Y: Tensor) -> dict[str, Tensor]:
    """Per-example classification metrics of the model: accuracy and margin."""
    correct = out.gather(1, Y[:, None])[:, 0]
    best_other = out.scatter(1, Y[:, None], float("-inf")).max(1).values
    return {"acc": (out.argmax(1) == Y).to(out.dtype), "margin": correct - best_other}


@dataclass(frozen=True, kw_only=True)
class Objective:
    """The loss `loss_fn` of `model` as a function of its params dict, on the examples it is given.
    - `bf16`: the model forward in bfloat16 (autocast)
    """

    model: nn.Module
    loss_fn: LossFn
    bf16: bool = False

    def example_losses(self, params: dict[str, Tensor], X: Tensor, Y: Tensor) -> tuple[Tensor, Tensor]:
        """The model on a batch: (per-example losses [B], outputs [B, C])"""
        with torch.autocast(X.device.type, torch.bfloat16, enabled=self.bf16):
            out: Tensor = functional_call(self.model, params, (X,))
        out = out.to(X.dtype)
        return self.loss_fn(out, Y), out

    def forward(
        self, params: dict[str, Tensor], X: Tensor, Y: Tensor, weights: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """The model on a batch: (mean loss, with the example `weights` [B] if given; outputs [B, C])"""
        losses, out = self.example_losses(params, X, Y)
        return (losses if weights is None else weights * losses).mean(), out

    @torch.no_grad()
    def eval_loss(self, params: dict[str, Tensor], loader: Loader, indices: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Mean loss and metrics of `params` on the examples `indices` of `loader`, accumulated batch by batch"""
        n = len(indices)
        loss, mean_metrics = torch.zeros((), device=loader.device), {}
        for X, Y in loader.iter_batches(indices):
            batch_loss, out = self.forward(params, X, Y)
            loss += batch_loss * len(X) / n
            for k, v in metrics(out, Y).items():
                mean_metrics[k] = mean_metrics.get(k, 0) + v.sum() / n
        return loss, mean_metrics

    def hvp_fn(self, params: dict[str, Tensor], X: Tensor, Y: Tensor) -> HVP:
        """`hvp_from_grads` of the mean loss of the batch (X, Y) at `params`"""
        params = {k: p.detach().requires_grad_() for k, p in params.items()}
        grads = torch.autograd.grad(self.forward(params, X, Y)[0], params, create_graph=True)
        return hvp_from_grads(grads, params)

    def eval_grads(self, params: dict[str, Tensor], eval_sets: Sequence[tuple[Tensor, Tensor]]) -> tuple[Tensor, dict[str, Tensor]]:
        """(f [#fns], df/d(params) {name: [#fns, *shape]}): f_i = mean loss on the examples (X_i, Y_i) of `eval_sets`, in one forward"""
        params = {k: p.detach().requires_grad_() for k, p in params.items()}
        device = next(iter(params.values())).device
        X, Y = (torch.cat(tensors).to(device) for tensors in zip(*eval_sets))
        losses = self.example_losses(params, X, Y)[0]
        means = torch.block_diag(*[losses.new_full((1, len(X_i)), 1 / len(X_i)) for X_i, _ in eval_sets])  # [#fns, #examples]
        grads = torch.autograd.grad(losses, params, means, is_grads_batched=True)
        return means @ losses.detach(), grads
