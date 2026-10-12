from dataclasses import dataclass
from functools import partial
from typing import Iterator

import torch
from torch import Tensor


class Dataset:
    """Classification dataset with int labels."""

    in_shape: tuple[int, ...]
    out_dim: int  # number of classes
    labels: Tensor  # [N] int64, on CPU

    def __len__(self) -> int:
        return len(self.labels)

    def get_batch(self, idx: Tensor) -> tuple[Tensor, Tensor]:
        """The batch (X, Y) of the examples `idx` (a CPU tensor)."""
        raise NotImplementedError


class Images(Dataset):
    """MNIST or CIFAR10 from torchvision, kept as uint8 [N, C, H, W] and normalized per batch."""

    stats = {"MNIST": ((0.1307,), (0.3081,)), "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))}

    def __init__(self, name: str, train: bool, root: str = "./data"):
        import torchvision.datasets

        data = getattr(torchvision.datasets, name)(root, train=train, download=True)
        X = torch.as_tensor(data.data)  # MNIST: [N, 28, 28]; CIFAR10: [N, 32, 32, 3]
        self.X = (X[:, None] if X.ndim == 3 else X.permute(0, 3, 1, 2)).contiguous()
        self.labels = torch.as_tensor(data.targets, dtype=torch.long)
        mean, std = self.stats[name]
        self.mean, self.std = torch.tensor(mean)[:, None, None], torch.tensor(std)[:, None, None]
        self.in_shape, self.out_dim = tuple(self.X.shape[1:]), 10

    def get_batch(self, idx: Tensor) -> tuple[Tensor, Tensor]:
        X = (self.X[idx].to(torch.get_default_dtype()) / 255 - self.mean) / self.std
        return X, self.labels[idx]


class Majority(Dataset):
    """Toy task: x ~ N(0, I), y = [sum(x) > 0] with ~2% label noise.
    Example i is generated from its own seed, so only the labels are stored."""

    def __init__(self, train: bool, size: int = 512, dim: int = 32, seed: int = 0):
        self.dim, self.seed = dim, 2 * seed + (0 if train else 1)
        self.in_shape, self.out_dim = (dim,), 2
        self.labels = torch.stack([self.example(i)[1] for i in range(size)])

    def example(self, i: int) -> tuple[Tensor, Tensor]:
        g = torch.Generator().manual_seed((self.seed << 32) + i)
        x = torch.randn(self.dim, generator=g)
        flip = torch.randn((), generator=g) > 2
        return x, torch.tensor(int((x.sum() > 0) != flip))

    def get_batch(self, idx: Tensor) -> tuple[Tensor, Tensor]:
        X, Y = zip(*(self.example(i) for i in idx.tolist()))
        return torch.stack(X), torch.stack(Y)


DATASETS = {"mnist": partial(Images, "MNIST"), "cifar10": partial(Images, "CIFAR10"), "majority": Majority}


@dataclass
class Loader:
    """Random batches deterministic in `step`."""

    dataset: Dataset
    batch_size: int = 32
    device: torch.device | str | None = None
    seed: int = 0

    def get(self, step: int) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (X, Y, indices) for `step`, on `device`."""
        g = torch.Generator().manual_seed((self.seed << 32) + step)
        idx = torch.randint(len(self.dataset), (self.batch_size,), generator=g)
        X, Y = self.dataset.get_batch(idx)
        X, Y, idx = (t.to(self.device, non_blocking=True) for t in (X, Y, idx))
        return X, Y, idx


@dataclass
class EvalLoader:
    """The fixed examples `indices` of `dataset` (all if None), in order, in batches on `device`."""

    dataset: Dataset
    indices: Tensor | None = None  # CPU; None -> all
    batch_size: int = 256
    device: torch.device | str | None = None

    def __post_init__(self):
        if self.indices is None:
            self.indices = torch.arange(len(self.dataset))

    def __len__(self) -> int:
        """Number of examples."""
        assert self.indices is not None
        return len(self.indices)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        assert self.indices is not None
        for i in range(0, len(self.indices), self.batch_size):
            X, Y = self.dataset.get_batch(self.indices[i : i + self.batch_size])
            yield X.to(self.device, non_blocking=True), Y.to(self.device, non_blocking=True)


@dataclass(frozen=True, kw_only=True)
class Subset:
    """The `index`-th random subset of `frac` * (dataset size) examples, all of class `label`
    (or of any class if None). Training runs drop it (weight 0), and evaluation functions are
    the mean loss over it.
    """

    frac: float
    label: int | None = None
    index: int = 0

    def mask(self, data: Dataset) -> Tensor:
        """Boolean mask [N] (on CPU), True on the selected examples."""
        idx = torch.arange(len(data)) if self.label is None else (data.labels == self.label).nonzero()[:, 0]
        n = round(self.frac * len(data))
        if n > len(idx):
            raise ValueError(f"{self.frac:.0%} of the {len(data)} examples is {n}, but label {self.label} has only {len(idx)}")
        perm = torch.randperm(len(idx), generator=torch.Generator().manual_seed(self.index))
        mask = torch.zeros(len(data), dtype=torch.bool)
        mask[idx[perm[:n]]] = True
        return mask

    @property
    def name(self) -> str:
        return f"{self.frac:.1%} of " + ("all labels" if self.label is None else f"label {self.label}") + f" #{self.index}"

    @property
    def dirname(self) -> str:
        return self.name.replace("% of ", "pct_").replace(" ", "_").replace("#", "")


@dataclass(frozen=True, kw_only=True)
class Perturbation:
    """The `index`-th random change of the per-example training weights w (1 = unchanged):
    - kind "drop": w = 0 on a random `alpha` fraction of the examples (of class `label` if given, see `Subset`)
    - kind "randn": w = 1 + N(0, alpha) (variance alpha) on every example: continuous in alpha, and
      ||w - 1||^2 = alpha * N like a drop of a fraction alpha
    """

    kind: str = "drop"
    alpha: float
    label: int | None = None
    index: int = 0

    def weights(self, data: Dataset) -> Tensor:
        """The weights [N] (on CPU)."""
        if self.kind == "drop":
            return (~Subset(frac=self.alpha, label=self.label, index=self.index).mask(data)).to(torch.get_default_dtype())
        assert self.kind == "randn" and self.label is None, self
        return 1 + self.alpha**0.5 * torch.randn(len(data), generator=torch.Generator().manual_seed(self.index))

    @property
    def group(self) -> str:
        """What the perturbation does, without the index (e.g. to color points)."""
        if self.kind == "randn":
            return f"randn {self.alpha:g}"
        return f"drop {self.alpha:.1%} of " + ("all labels" if self.label is None else f"label {self.label}")

    @property
    def dirname(self) -> str:
        return self.group.replace(" of ", "_").replace(" ", "_").replace("%", "pct") + f"_{self.index}"
