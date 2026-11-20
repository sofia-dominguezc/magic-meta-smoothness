"""The loss as a function of the model parameters on given examples (`Objective`), its derivatives (gradients, Hessian-vector
products), and the losses and metrics."""

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp, vmap


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


def _forward_hvp(forward_fn: Callable, params: dict[str, Tensor], V: dict[str, Tensor]) -> dict[str, Tensor]:
    """H V for V {name: [m, *shape]}, forward-over-reverse"""
    grad_fn = grad(lambda p: forward_fn(p)[0])
    return vmap(lambda v: jvp(grad_fn, (params,), (v,))[1])(V)


_compiled_forward_hvp = torch.compile(_forward_hvp)  # (module-level: one cache for every step's `forward_fn`)


def hvp_from_forward(forward_fn: Callable, params: dict[str, Tensor], compile: bool = False) -> HVP:
    """V -> H V of the loss `forward_fn(params)[0]` by torch.func, without a graph (`compile`: torch.compiled, once per number of vectors)"""
    params = {k: p.detach() for k, p in params.items()}
    hvp = _compiled_forward_hvp if compile else _forward_hvp

    def hvp_fn(V: dict[str, Tensor], create_graph: bool = False) -> dict[str, Tensor]:
        assert not create_graph
        if next(iter(V.values())).ndim > next(iter(params.values())).ndim:
            return hvp(forward_fn, params, V)
        return {k: Hv[0] for k, Hv in hvp(forward_fn, params, {k: v[None] for k, v in V.items()}).items()}

    return hvp_fn


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
        # (a dot product: `(weights * losses).mean()` doesn't torch.compile under nested forward mode)
        return losses.mean() if weights is None else torch.dot(weights, losses) / len(losses), out

    @torch.no_grad()
    def eval_loss(self, params: dict[str, Tensor], batches: Iterable[tuple[Tensor, Tensor]]) -> tuple[Tensor, dict[str, Tensor]]:
        """Mean loss and metrics of `params` over the examples of `batches` (X, Y)"""
        n_examples = 0
        sums: dict[str, Tensor] = {}
        for X, Y in batches:
            losses, out = self.example_losses(params, X, Y)
            for k, v in {"loss": losses, **metrics(out, Y)}.items():
                sums[k] = sums.get(k, 0) + v.sum()
            n_examples += len(X)
        means = {k: total / n_examples for k, total in sums.items()}
        return means.pop("loss"), means

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
