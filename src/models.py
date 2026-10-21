"""Models in the spectral form of μP (Yang, Simon & Bernstein 2023, "A spectral condition for feature learning"): layers
store raw tensors with O(1) operator norm (`logging.op_norm`) behind fixed forward multipliers (`Linear`), so that one
learning rate serves every raw tensor and transfers across widths (`OptimState.precond`). Normalization layers are
parameter-free: the training state is the parameters. Seed with torch.manual_seed before creating a model."""

import math
from dataclasses import dataclass, replace
from typing import Callable, ClassVar, Iterator, Iterable, Mapping, Self

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def mu_init_(layer: nn.Linear | nn.Conv2d, readout: bool = False):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)
    nn.init.normal_(layer.weight, std=(fan_out * (fan_in if readout else 1)) ** -0.5)
    if hasattr(layer, "bias"):
        nn.init.zeros_(layer.bias)


class Linear(nn.Linear):
    """W = sqrt(out_dim / in_dim) * w and b = sqrt(out_dim) * b_raw, with w ~ N(0, 1/out_dim) (readout: N(0, 1/(in_dim out_dim)))
    and b_raw = 0: μP's per-layer learning rates became these multipliers (abc-symmetry)."""

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

    def __init__(self, in_shape: tuple[int, ...], out_dim: int, width: int = 32, blocks: int = 1):
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


def dict_zip[T](*dicts: Mapping[str, T]) -> dict[str, tuple[T, ...]]:
    """e.g. {"a": 1, "b": 2}, {"a": 3, "b": 4} -> {"a": (1, 3), "b": (2, 4)}"""
    assert all(set(dicts[0]) == set(d) for d in dicts)
    return {key: tuple(d[key] for d in dicts) for key in dicts[0]}


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
        return self.lr * (self.FINAL_FRACTION + (1 - self.FINAL_FRACTION) * (1 + math.cos(math.pi * progress)) / 2)


@dataclass(kw_only=True)
class OptimState:
    """All model + optimizer state tensors (the dict fields named in `groups`) and the step on them."""

    groups: ClassVar[tuple[str, ...]] = ("params",)
    params: dict[str, Tensor]  # model parameters
    lr: LRSchedule
    weight_decay: float = 0.0

    def __post_init__(self):
        if isinstance(self.lr, dict):  # saved by `asdict`
            self.lr = LRSchedule(**self.lr)

    def __getitem__(self, group: str) -> dict[str, Tensor]:
        assert group in self.groups
        return getattr(self, group)

    def __iter__(self) -> Iterator[Tensor]:
        return (t for group in self.groups for t in self[group].values())

    def update(self, fn: Callable[[Tensor], Tensor]):
        """Replace each tensor by `fn(tensor)`, in place"""
        for group in self.groups:
            setattr(self, group, {k: fn(t) for k, t in self[group].items()})

    def apply(self, fn: Callable[[Tensor], Tensor]) -> Self:
        """A copy of the state with each tensor replaced by `fn(tensor)`"""
        new = replace(self)
        new.update(fn)
        return new

    def unflatten(self, tensors: Iterable[Tensor]) -> Self:
        """Inverse of iterating: a state like this one holding `tensors` instead"""
        it = iter(tensors)
        return self.apply(lambda _: next(it))

    def step(self, t: int, grads: dict[str, Tensor]) -> Self:
        """Update the state tensors by step `t`, in place (the entries are replaced, not the tensors). Returns self."""
        raise NotImplementedError

    def precond(self, t: int) -> Mapping[str, Tensor | float]:
        """Per-element step size P of step `t` (the update is P * grad, or P * m for Adam): lr(t) for SGD."""
        return {k: self.lr(t) for k in self.params}

    @property
    def stability_threshold(self) -> float:
        """The EoS condition is `max_eig(sqrt(P) H sqrt(P)) = stability_threshold`"""
        raise NotImplementedError


@dataclass(kw_only=True)
class SGDState(OptimState):
    def step(self, t: int, grads: dict[str, Tensor]) -> Self:
        P = self.precond(t)
        self.params = {
            k: p - P[k] * (g + self.weight_decay * p)
            for k, (p, g) in dict_zip(self.params, grads).items()
        }
        return self

    @property
    def stability_threshold(self) -> float:
        return 2.0


@dataclass(kw_only=True)
class SGDMomentumState(SGDState):
    groups = ("params", "buffer")
    momentum: float = 0.9
    buffer: dict[str, Tensor] | None = None  # None -> zeros

    def __post_init__(self):
        super().__post_init__()
        if self.buffer is None:
            self.buffer = {k: torch.zeros_like(p) for k, p in self.params.items()}

    def step(self, t: int, grads: dict[str, Tensor]) -> Self:
        assert self.buffer is not None
        P = self.precond(t)
        self.buffer = {
            k: self.momentum * buf + g + self.weight_decay * p
            for k, (p, buf, g) in dict_zip(self.params, self.buffer, grads).items()
        }
        self.params = {
            k: p - P[k] * buf
            for k, (p, buf) in dict_zip(self.params, self.buffer).items()
        }
        return self

    @property
    def stability_threshold(self) -> float:
        return 2 * (1 + self.momentum)


@dataclass(kw_only=True)
class AdamState(OptimState):
    """Adam as in torch.optim.Adam (L2 weight decay), with the μP learning rates `lrs`."""

    groups = ("params", "m", "v")
    lr_mults: dict[str, float] | None = None  # per-tensor multipliers of lr (None: μP, 1/sqrt(numel): a step moves each raw
    # tensor by O(lr) in operator norm like SGD)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    m: dict[str, Tensor] | None = None  # None -> zeros
    v: dict[str, Tensor] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.lr_mults is None:
            self.lr_mults = {k: p.numel() ** -0.5 for k, p in self.params.items()}
        if self.m is None:
            self.m = {k: torch.zeros_like(p) for k, p in self.params.items()}
        if self.v is None:
            self.v = {k: torch.zeros_like(p) for k, p in self.params.items()}

    def _step_sizes(self, v: dict[str, Tensor], t: int) -> dict[str, Tensor]:
        """Per-element step size of step t (the (t + 1)-th, for the bias correction, training from step 0):
        lr(t) * lr_mults / (bias-corrected sqrt(v) + eps)."""
        assert self.lr_mults is not None
        b1, b2, n, lr = *self.betas, t + 1, self.lr(t)
        return {
            k: lr * self.lr_mults[k] / (1 - b1**n) / (v_k**0.5 / (1 - b2**n) ** 0.5 + self.eps)
            for k, v_k in v.items()
        }

    def step(self, t: int, grads: dict[str, Tensor]) -> Self:
        assert self.m is not None and self.v is not None
        b1, b2 = self.betas
        grads = {k: g + self.weight_decay * p for k, (g, p) in dict_zip(grads, self.params).items()}
        new_m = {k: b1 * m + (1 - b1) * g for k, (m, g) in dict_zip(self.m, grads).items()}
        new_v = {k: b2 * v + (1 - b2) * g * g for k, (v, g) in dict_zip(self.v, grads).items()}
        sizes = self._step_sizes(new_v, t)
        self.params = {k: p - size * m for k, (p, m, size) in dict_zip(self.params, new_m, sizes).items()}
        self.m, self.v = new_m, new_v
        return self

    def precond(self, t: int) -> dict[str, Tensor]:
        """With this state's v (exact on the state after step t)."""
        assert self.v is not None
        return self._step_sizes(self.v, t)

    @property
    def stability_threshold(self) -> float:
        """Cohen et al. 2022, "Adaptive gradient methods at the edge of stability": 38 for beta1 = 0.9."""
        return 2 * (1 + self.betas[0]) / (1 - self.betas[0])
