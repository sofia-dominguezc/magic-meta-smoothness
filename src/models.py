"""Models in the spectral form of μP (Yang, Simon & Bernstein 2023, "A spectral condition for feature learning"): layers
store raw tensors with O(1) operator norm (`logging.op_norm`) behind fixed forward multipliers (`Linear`), so that one
learning rate serves every raw tensor and transfers across widths (`optimizers.OptimState.precond`). Normalization layers are
parameter-free: the training state is the parameters. Seed with torch.manual_seed before creating a model."""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def mu_init_(layer: nn.Linear | nn.Conv2d, readout: bool = False):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)
    nn.init.normal_(layer.weight, std=(fan_out * (fan_in if readout else 1)) ** -0.5)
    if layer.bias is not None:
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
    N(0, 1/width), used as sqrt(width) * pos), heads of `head_dim` dims, mean pooling and a linear readout."""

    def __init__(self, in_shape: tuple[int, ...], out_dim: int, width: int = 96, depth: int = 3, head_dim: int = 32, patch: int = 4):
        super().__init__()
        C, H, W = in_shape
        self.patch = patch
        self.embed = Linear(C * patch * patch, width)
        self.pos = nn.Parameter(torch.randn((H // patch) * (W // patch), width) / width**0.5)
        self.blocks = nn.ModuleList(TransformerBlock(width, width // head_dim) for _ in range(depth))
        self.out = Linear(width, out_dim, readout=True)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        p = self.patch
        x = x.view(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)
        x = self.embed(x) + self.pos.shape[1] ** 0.5 * self.pos
        for block in self.blocks:
            x = block(x)
        return self.out(norm(x, -1).mean(1))


MODELS: dict[str, type[nn.Module]] = {"mlp": MLP, "cnn": ResNet, "vit": ViT}
