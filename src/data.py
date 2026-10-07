from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor


class Dataset:
    out_dim: int

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def label(self, i: int) -> int:
        """Class of example i, used to choose subsets by label."""
        raise NotImplementedError

    def metrics(self, out: Tensor, Y: Tensor) -> dict[str, Tensor]:
        """Per-example metrics of the model output, for logging."""
        return {}

    def get_batch(self, idx: Tensor) -> tuple[Tensor, Tensor]:
        """The batch (X, Y) of the examples `idx` (a CPU tensor)."""
        X, Y = zip(*(self[i] for i in idx.tolist()))
        return torch.stack(X), torch.stack(Y)


class MNIST(Dataset):
    out_dim = 10

    def __init__(self, train: bool, root: str = "./data"):
        from torchvision.datasets import MNIST
        from torchvision.transforms import Compose, ToTensor, Normalize

        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        self.data = MNIST(root, train=train, download=True, transform=transform)

    def __len__(self) -> int:
        return len(self.data)

    def label(self, i: int) -> int:
        return int(self.data.targets[i])

    def metrics(self, out: Tensor, Y: Tensor) -> dict[str, Tensor]:
        correct = out.gather(1, Y[:, None])[:, 0]
        best_other = out.scatter(1, Y[:, None], float("-inf")).max(1).values
        return {"acc": (out.argmax(1) == Y).float(), "margin": correct - best_other}

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        x, y = self.data[i]  # [1, 28, 28], int
        return x, torch.tensor(y)


class Majority(Dataset):
    """Toy task: y = sign(sum(x)) with ~2% label noise (use with the MSE loss).
    Example i is generated from its own seed, so nothing is stored."""

    out_dim = 1

    def __init__(self, train: bool, size: int = 512, dim: int = 32, seed: int = 0):
        self.size, self.dim, self.seed = size, dim, 2 * seed + (0 if train else 1)

    def __len__(self) -> int:
        return self.size

    def label(self, i: int) -> int:
        return int(self[i][1].item() > 0)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        g = torch.Generator().manual_seed((self.seed << 32) + i)
        x = torch.randn(self.dim, generator=g)
        y = torch.where(x.sum() > 0, 1.0, -1.0).to(x.dtype)
        if torch.randn((), generator=g) > 2:
            y = -y
        return x, y[None]

    def metrics(self, out: Tensor, Y: Tensor) -> dict[str, Tensor]:
        return {"acc": ((out > 0) == (Y > 0)).float().mean(1), "margin": (out * Y).mean(1)}


DATASETS = {"mnist": MNIST, "majority": Majority}


@dataclass
class Loader:
    """Random batches deterministic in `step`, with per-example weights"""

    dataset: Dataset
    batch_size: int = 32
    device: torch.device | str | None = None
    weights: Tensor | None = None  # [N] on CPU; None -> all ones
    seed: int = 0

    def __post_init__(self):
        if self.weights is None:
            self.weights = torch.ones(len(self.dataset))

    def _to_device(self, t: Tensor) -> Tensor:
        return t.to(self.device, non_blocking=True)

    def get(self, step: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns (X, Y, indices, weights) for `step`, on `device`."""
        assert self.weights is not None
        g = torch.Generator().manual_seed((self.seed << 32) + step)
        idx = torch.randint(len(self.dataset), (self.batch_size,), generator=g)
        X, Y = self.dataset.get_batch(idx)
        X, Y, idx, w = map(self._to_device, (X, Y, idx, self.weights[idx]))
        return X, Y, idx, w


@dataclass
class EvalLoader:
    """The fixed examples `indices` of `dataset` (all if None), in order, in batches on `device`."""

    dataset: Dataset
    indices: Tensor | None = None  # CPU; None -> all
    batch_size: int = 32
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
    """A deterministic random subset: `frac` of the examples of class `label` (or of
    all examples if None), chosen with `seed`. Training runs drop it (weight 0), and
    evaluation functions are the mean loss over it.
    """

    frac: float
    label: int | None = None
    seed: int = 0

    def mask(self, data: Dataset) -> Tensor:
        """Boolean mask [N] (on CPU), True on the selected examples."""
        if self.label is None:
            idx = torch.arange(len(data))
        else:  # one pass over the labels, which aren't stored
            idx = torch.tensor([i for i in range(len(data)) if data.label(i) == self.label], dtype=torch.long)
        perm = torch.randperm(len(idx), generator=torch.Generator().manual_seed(self.seed))
        mask = torch.zeros(len(data), dtype=torch.bool)
        mask[idx[perm[: round(self.frac * len(idx))]]] = True
        return mask
