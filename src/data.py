from dataclasses import dataclass, field
from functools import partial
from typing import Iterator

import torch
from torch import Tensor


class Dataset:
    """Classification dataset with int labels. Subclasses set `in_shape`, `out_dim`, `labels`, `ids` and implement
    `examples`."""

    in_shape: tuple[int, ...]
    out_dim: int  # number of classes
    labels: Tensor  # [N] int64, on CPU
    ids: Tensor  # [N] the examples' original indices (arange, unless `take`n)

    def __len__(self) -> int:
        return len(self.labels)

    def take(self, n: int, seed: int = 0):
        """Keep a random subset of `n` examples, in place: they become the examples 0, ..., n - 1 (in their
        original order; `ids` maps them to the originals)."""
        keep = torch.randperm(len(self), generator=torch.Generator().manual_seed(seed))[:n].sort().values
        self.ids, self.labels = self.ids[keep], self.labels[keep]

    def examples(self, idx: Tensor) -> tuple[Tensor, Tensor]:
        """(X, Y) of the examples `idx` (a CPU tensor), on CPU, X in the default dtype at call time."""
        raise NotImplementedError

    def subset(self, n: int, index: int = 0) -> Tensor:
        """The `index`-th random subset of `n` examples: sorted indices [n] (on CPU). An eval function is the mean loss
        over a subset of the test set."""
        if n > len(self):
            raise ValueError(f"{n} examples asked of {len(self)}")
        perm = torch.randperm(len(self), generator=torch.Generator().manual_seed(index))
        return perm[:n].sort().values

    def weights(self, kind: str, alpha: float, index: int = 0) -> Tensor:
        """The `index`-th random perturbation of the per-example training weights w [N] (1 = unchanged, on CPU):
        - "drop": w = 0 on a random `alpha` fraction of the examples (`subset`)
        - "randn": w = 1 + N(0, alpha) (variance alpha) on every example: continuous in alpha, and
          ||w - 1||^2 = alpha * N like a drop of a fraction alpha
        """
        if kind == "drop":
            w = torch.ones(len(self))
            w[self.subset(round(alpha * len(self)), index)] = 0
            return w
        assert kind == "randn", kind
        return 1 + alpha**0.5 * torch.randn(len(self), generator=torch.Generator().manual_seed(index))


class Images(Dataset):
    """MNIST or CIFAR10 from torchvision, kept as uint8 [N, C, H, W] and normalized per batch."""

    stats = {"MNIST": ((0.1307,), (0.3081,)), "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))}

    def __init__(self, name: str, train: bool, root: str = "./data"):
        import torchvision.datasets

        data = getattr(torchvision.datasets, name)(root, train=train, download=True)
        X = torch.as_tensor(data.data)  # MNIST: [N, 28, 28]; CIFAR10: [N, 32, 32, 3]
        self.X = (X[:, None] if X.ndim == 3 else X.permute(0, 3, 1, 2)).contiguous()
        self.labels = torch.as_tensor(data.targets, dtype=torch.long)
        self.ids = torch.arange(len(self.labels))
        mean, std = self.stats[name]
        self.mean, self.std = torch.tensor(mean, dtype=torch.float64)[:, None, None], torch.tensor(std, dtype=torch.float64)[:, None, None]
        self.in_shape, self.out_dim = tuple(self.X.shape[1:]), 10

    def examples(self, idx: Tensor) -> tuple[Tensor, Tensor]:
        dtype = torch.get_default_dtype()
        X = (self.X[self.ids[idx]].to(dtype) / 255 - self.mean.to(dtype)) / self.std.to(dtype)
        return X, self.labels[idx]


class Majority(Dataset):
    """Toy task: x ~ N(0, I), y = [sum(x) > 0] with ~2% label noise.
    Example i is generated from its own seed, so only the labels are stored."""

    def __init__(self, train: bool, size: int = 512, dim: int = 32, seed: int = 0):
        self.dim, self.seed = dim, 2 * seed + (0 if train else 1)
        self.in_shape, self.out_dim = (dim,), 2
        self.labels = torch.stack([self.example(i)[1] for i in range(size)])
        self.ids = torch.arange(size)

    def example(self, i: int) -> tuple[Tensor, Tensor]:
        """Generated in float64 (the float32 and float64 samplers draw different numbers from the same seed, and the
        labels must match the inputs whatever the default dtype)."""
        g = torch.Generator().manual_seed(self.seed * 2**20 + i)  # (the CPU generator keeps the low 32 bits of a seed)
        x = torch.randn(self.dim, generator=g, dtype=torch.float64)
        flip = torch.randn((), generator=g, dtype=torch.float64) > 2
        return x, torch.tensor(int((x.sum() > 0) != flip))

    def examples(self, idx: Tensor) -> tuple[Tensor, Tensor]:
        X, Y = zip(*(self.example(i) for i in self.ids[idx].tolist()))
        return torch.stack(X).to(torch.get_default_dtype()), torch.stack(Y)


DATASETS = {"mnist": partial(Images, "MNIST"), "cifar10": partial(Images, "CIFAR10"), "majority": Majority}


@dataclass
class Loader:
    """Batches of `dataset` on `device`: the training batches (deterministic in the step), or given examples in order."""

    dataset: Dataset
    batch_size: int = 32
    device: torch.device | str | None = None
    seed: int = 0
    _perm: tuple[int, Tensor] | None = field(default=None, repr=False)  # (epoch, its permutation), cached

    def batch_indices(self, step: int) -> Tensor:
        """The examples (CPU indices) of the training batch of `step`: epochs of len // batch_size batches without
        replacement (a permutation per epoch, seeded by `seed` and the epoch; its last len % batch_size examples are skipped)"""
        n_batches = len(self.dataset) // self.batch_size
        assert n_batches > 0, f"batch size {self.batch_size} > {len(self.dataset)} examples"
        epoch, i = divmod(step, n_batches)
        if self._perm is None or self._perm[0] != epoch:
            g = torch.Generator().manual_seed(self.seed * 2**20 + epoch)  # (the CPU generator keeps the low 32 bits)
            self._perm = (epoch, torch.randperm(len(self.dataset), generator=g))
        return self._perm[1][i * self.batch_size : (i + 1) * self.batch_size]

    def get_batch(self, step: int) -> tuple[Tensor, Tensor, Tensor]:
        """The training batch of `step` (`batch_indices`) on `device`: (X, Y, indices)."""
        idx = self.batch_indices(step)
        X, Y = self.dataset.examples(idx)
        X, Y, idx = (t.to(self.device, non_blocking=True) for t in (X, Y, idx))
        return X, Y, idx

    def iter_batches(self, indices: Tensor) -> Iterator[tuple[Tensor, Tensor]]:
        """The examples `indices` (CPU), in order, in batches of `batch_size`: (X, Y)."""
        for i in range(0, len(indices), self.batch_size):
            X, Y = self.dataset.examples(indices[i : i + self.batch_size])
            yield X.to(self.device, non_blocking=True), Y.to(self.device, non_blocking=True)
