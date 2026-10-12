"""Models in the maximal update parametrization (μP), in its "spectral" form (Yang, Simon & Bernstein 2023,
"A spectral condition for feature learning"). A layer with weight W: fan_in -> fan_out and bias b stores raw
tensors (w, b_raw) and computes
    W = sqrt(fan_out / fan_in) * w,    b = sqrt(fan_out) * b_raw
- init: w ~ N(0, 1/fan_out), so that W has the LeCun variance 1/fan_in (readout: N(0, 1/(fan_in fan_out)),
  i.e. μP's 1/fan_in^2); b_raw = 0
- this is the usual μP table up to the abc-symmetry (same training dynamics): the per-layer SGD learning rates
  lr * fan_out / fan_in become forward multipliers, and one lr is used for all raw tensors
- consequences: the optimizer state stays simple, the lr transfers across widths, the Hessian of the raw
  parameters is the one whose top eigenvalue sets the stability of a step (2/lr for SGD), and every raw tensor
  (parameter, gradient, update) has an O(1) operator norm in every layer (`utils.op_norm`)
- Adam needs per-tensor lr multipliers instead: `adam_lr_mults`
- normalization layers are parameter-free, so that the whole training state is the parameters
Seed with torch.manual_seed before creating a model."""

import math
from dataclasses import dataclass, replace
from typing import Callable, ClassVar, Iterator, Self

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def mu_init_(layer: nn.Linear | nn.Conv2d, readout: bool = False):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)
    nn.init.normal_(layer.weight, std=(fan_out * (fan_in if readout else 1)) ** -0.5)
    nn.init.zeros_(layer.bias)


class Linear(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, readout: bool = False):
        super().__init__(in_dim, out_dim)
        self.mult = (out_dim / in_dim) ** 0.5
        mu_init_(self, readout)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.mult * self.weight, self.out_features**0.5 * self.bias)


class Conv2d(nn.Conv2d):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.mult = (out_ch / in_ch) ** 0.5
        mu_init_(self)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.mult * self.weight, self.out_channels**0.5 * self.bias)


def norm(x: Tensor, dims: int | tuple[int, ...], eps: float = 1e-5) -> Tensor:
    """Parameter-free layer norm over `dims`."""
    x = x - x.mean(dims, keepdim=True)
    return x * torch.rsqrt(x.pow(2).mean(dims, keepdim=True) + eps)


class MLP(nn.Module):
    def __init__(self, in_shape: tuple[int, ...], out_dim: int, width: int = 512, depth: int = 2):
        super().__init__()
        dims = [math.prod(in_shape)] + [width] * depth
        self.hidden = nn.ModuleList(Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.out = Linear(width, out_dim, readout=True)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        for layer in self.hidden:
            x = F.gelu(layer(x))
        return self.out(x)


class Block(nn.Module):
    """Pre-activation residual block: x + conv(gelu(norm(conv(gelu(norm(x)))))), with a 1x1 shortcut if the shape changes."""

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.conv1 = Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = Conv2d(in_ch, out_ch, 1, stride=stride) if stride != 1 or in_ch != out_ch else None

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(norm(x, (1, 2, 3)))
        skip = x if self.shortcut is None else self.shortcut(h)
        return skip + self.conv2(F.gelu(norm(self.conv1(h), (1, 2, 3))))


class ResNet(nn.Module):
    """Pre-activation ResNet for small images: a 3x3 stem, 3 stages of `blocks` residual blocks with
    width, 2 * width, 4 * width channels (strides 1, 2, 2), global average pooling and a linear readout."""

    def __init__(self, in_shape: tuple[int, ...], out_dim: int, width: int = 64, blocks: int = 1):
        super().__init__()
        self.stem = Conv2d(in_shape[0], width, 3, padding=1)
        self.blocks = nn.ModuleList()
        in_ch = width
        for stage, out_ch in enumerate([width, 2 * width, 4 * width]):
            for block in range(blocks):
                self.blocks.append(Block(in_ch, out_ch, stride=2 if stage > 0 and block == 0 else 1))
                in_ch = out_ch
        self.out = Linear(in_ch, out_dim, readout=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.out(F.gelu(norm(x, (1, 2, 3))).mean((2, 3)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = Linear(dim, 3 * dim)
        self.proj = Linear(dim, dim)
        self.fc1 = Linear(dim, 4 * dim)
        self.fc2 = Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(norm(x, -1)).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)  # [B, h, N, d]
        attn = (q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5).softmax(-1)
        x = x + self.proj((attn @ v).transpose(1, 2).flatten(2))
        return x + self.fc2(F.gelu(self.fc1(norm(x, -1))))


class ViT(nn.Module):
    """Pre-norm ViT with `patch` x `patch` patches, learned positional embeddings (a bias-like raw tensor:
    N(0, 1/dim), used as sqrt(dim) * pos), heads of `head_dim` dims, mean pooling and a linear readout."""

    def __init__(self, in_shape: tuple[int, ...], out_dim: int, dim: int = 192, depth: int = 4, head_dim: int = 32, patch: int = 4):
        super().__init__()
        C, H, W = in_shape
        self.patch = patch
        self.embed = Linear(C * patch * patch, dim)
        self.pos = nn.Parameter(torch.randn((H // patch) * (W // patch), dim) / dim**0.5)
        self.blocks = nn.ModuleList(TransformerBlock(dim, dim // head_dim) for _ in range(depth))
        self.out = Linear(dim, out_dim, readout=True)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        p = self.patch
        x = x.view(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)
        x = self.embed(x) + self.pos.shape[1] ** 0.5 * self.pos
        for block in self.blocks:
            x = block(x)
        return self.out(norm(x, -1).mean(1))


MODELS: dict[str, type[nn.Module]] = {"mlp": MLP, "cnn": ResNet, "vit": ViT}


def adam_lr_mults(model: nn.Module) -> dict[str, float]:
    """μP learning rate multipliers for Adam on the raw tensors (for SGD they are all 1 in this parametrization):
    1/sqrt(fan_in * fan_out) for weights, 1/sqrt(fan_out) for biases and positional embeddings, so that an Adam
    step (entries of size ~lr) moves each raw tensor by O(lr) in operator norm, like an SGD step."""
    mults = {}
    for name, p in model.named_parameters():
        if p.ndim == 1 or name.endswith("pos"):
            mults[name] = p.shape[-1] ** -0.5
        else:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(p)
            mults[name] = (fan_in * fan_out) ** -0.5
    return mults


def dict_zip[T](*dicts: dict[str, T]) -> dict[str, tuple[T, ...]]:
    """e.g. {"a": 1, "b": 2}, {"a": 3, "b": 4} -> {"a": (1, 3), "b": (2, 4)}"""
    return {key: tuple(d[key] for d in dicts) for key in dicts[0]}


@dataclass(kw_only=True)
class OptimState:
    """All model + optimizer state tensors: the dict fields named in `groups`."""

    groups: ClassVar[tuple[str, ...]] = ("params",)
    params: dict[str, Tensor]  # model parameters

    @property
    def param_groups(self) -> dict[str, dict[str, Tensor]]:
        return {group: getattr(self, group) for group in self.groups}

    def __iter__(self) -> Iterator[Tensor]:
        return (t for group in self.param_groups.values() for t in group.values())

    def apply(self, fn: Callable[[Tensor], Tensor]) -> Self:
        """Apply `fn` to each state tensor. Returns a new OptimState."""
        new_fields = {
            group_name: {name: fn(tensor) for name, tensor in group.items()}
            for group_name, group in self.param_groups.items()
        }
        return replace(self, **new_fields)

    def unflatten(self, tensors: list[Tensor]) -> Self:
        """Inverse of iterating: a state like this one holding `tensors` instead."""
        it = iter(tensors)
        return self.apply(lambda _: next(it))

    def step(self, grads: dict[str, Tensor]) -> Self:
        """Update state tensors. Returns a new OptimState."""
        raise NotImplementedError

    @property
    def precond(self) -> dict[str, Tensor] | None:
        """Per-element multiplier of the gradient in the update, up to `lr` (None: 1). The Hessian
        eigenvalues to compare with `stability_threshold` are those of the preconditioned Hessian."""
        return None

    @property
    def stability_threshold(self) -> float:
        """Top (preconditioned) Hessian eigenvalue above which the step is unstable (edge of stability)."""
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

    @property
    def stability_threshold(self) -> float:
        return 2 / self.lr


@dataclass(kw_only=True)
class SGDMomentumState(SGDState):
    groups = ("params", "buffer")
    momentum: float = 0.9
    buffer: dict[str, Tensor] | None = None  # None -> zeros

    def __post_init__(self):
        if self.buffer is None:
            self.buffer = {k: torch.zeros_like(p) for k, p in self.params.items()}

    def step(self, grads: dict[str, Tensor]) -> Self:
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

    @property
    def stability_threshold(self) -> float:
        return 2 * (1 + self.momentum) / self.lr


@dataclass(kw_only=True)
class AdamState(OptimState):
    """Adam as in torch.optim.Adam (L2 weight decay), with per-parameter learning rate
    multipliers `lr_mults` (None: 1; `models.adam_lr_mults` for μP)."""

    groups = ("params", "m", "v")
    lr: float
    lr_mults: dict[str, float] | None = None
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    t: int = 0  # steps taken
    m: dict[str, Tensor] | None = None  # None -> zeros
    v: dict[str, Tensor] | None = None

    def __post_init__(self):
        if self.lr_mults is None:
            self.lr_mults = {k: 1.0 for k in self.params}
        if self.m is None:
            self.m = {k: torch.zeros_like(p) for k, p in self.params.items()}
        if self.v is None:
            self.v = {k: torch.zeros_like(p) for k, p in self.params.items()}

    def _step_sizes(self, v: dict[str, Tensor], t: int) -> dict[str, Tensor]:
        """Per-element multiplier of m in the update at step t, up to lr: mult / (bias-corrected sqrt(v) + eps)."""
        assert self.lr_mults is not None
        b1, b2 = self.betas
        return {k: self.lr_mults[k] / (1 - b1**t) / (v_k.sqrt() / (1 - b2**t) ** 0.5 + self.eps) for k, v_k in v.items()}

    def step(self, grads: dict[str, Tensor]) -> Self:
        assert self.m is not None and self.v is not None
        b1, b2 = self.betas
        t = self.t + 1
        grads = {k: g + self.weight_decay * p for k, (g, p) in dict_zip(grads, self.params).items()}
        new_m = {k: b1 * m + (1 - b1) * g for k, (m, g) in dict_zip(self.m, grads).items()}
        new_v = {k: b2 * v + (1 - b2) * g * g for k, (v, g) in dict_zip(self.v, grads).items()}
        new_params = {
            k: p - self.lr * size * m
            for k, (p, m, size) in dict_zip(self.params, new_m, self._step_sizes(new_v, t)).items()
        }
        return replace(self, params=new_params, m=new_m, v=new_v, t=t)

    @property
    def precond(self) -> dict[str, Tensor]:
        assert self.v is not None
        return self._step_sizes(self.v, max(self.t, 1))

    @property
    def stability_threshold(self) -> float:
        """Cohen et al. 2022, "Adaptive gradient methods at the edge of stability": 38/lr for beta1 = 0.9."""
        return 2 * (1 + self.betas[0]) / ((1 - self.betas[0]) * self.lr)
