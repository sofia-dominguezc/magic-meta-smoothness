"""Helpers on params dicts {name: tensor}: Hessian-vector products, `dict_zip`, `flatten` / `unflatten`."""

from typing import Callable, Mapping, Protocol

import torch
from torch import Tensor
from torch.func import grad, jvp, vmap


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


def dict_zip[T](*dicts: Mapping[str, T]) -> dict[str, tuple[T, ...]]:
    """e.g. {"a": 1, "b": 2}, {"a": 3, "b": 4} -> {"a": (1, 3), "b": (2, 4)}"""
    assert all(set(dicts[0]) == set(d) for d in dicts)
    return {key: tuple(d[key] for d in dicts) for key in dicts[0]}


def flatten(d: dict[str, Tensor], batched: bool = True) -> Tensor:
    """{name: [m, *shape]} -> [#params, m] (the batch as columns); not batched: {name: [*shape]} -> [#params]"""
    return torch.cat([v.flatten(1) for v in d.values()], dim=1).T if batched else torch.cat([v.flatten() for v in d.values()])


def unflatten(V: Tensor, shapes: Mapping[str, torch.Size]) -> dict[str, Tensor]:
    """Inverse of `flatten`: [#params, m] -> {name: [m, *shapes[name]]}, [#params] -> {name: [*shapes[name]]}"""
    return {
        name: v.T.reshape(-1, *shape) if V.ndim == 2 else v.reshape(shape)
        for v, (name, shape) in zip(V.split([shape.numel() for shape in shapes.values()]), shapes.items())
    }
