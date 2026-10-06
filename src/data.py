from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor


def classification_metrics(out: Tensor, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    y = batch["y"]
    correct = out.gather(1, y[:, None])[:, 0]
    best_other = out.scatter(1, y[:, None], float("-inf")).max(1).values
    return {"acc": (out.argmax(1) == y).float(), "margin": correct - best_other}


def sign_metrics(out: Tensor, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    y = batch["y"]
    return {"acc": ((out > 0) == (y > 0)).float().mean(1), "margin": (out * y).mean(1)}


@dataclass
class Dataset:
    """A dataset held in memory on one device. Each tensor is [N, ...]; "x" is the
    model input and the loss can use any of the others (e.g. "y").
    """

    # TODO: storing the dataset in memory is the worst idea i've ever seen
    # batches and dataloaders exist for a reason
    tensors: dict[str, Tensor]
    labels: Tensor  # [N] integer classes on CPU, used to choose subsets by label
    out_dim: int
    metrics: Callable[[Tensor, dict[str, Tensor]], dict[str, Tensor]]  # (output, batch) -> per-example metrics, for logging

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: Tensor) -> dict[str, Tensor]:
        return {k: v[idx] for k, v in self.tensors.items()}

    @property
    def device(self) -> torch.device:
        return self.tensors["x"].device


def mnist(train: bool, device: torch.device | str, root: str = "./data") -> Dataset:
    cache = Path(root) / "MNIST" / f"{'train' if train else 'test'}_tensors.pt"
    if cache.exists():
        raw = torch.load(cache, weights_only=True)
    else:
        from torchvision import datasets

        ds = datasets.MNIST(root, train=train, download=True)
        raw = {"x": ds.data.clone(), "y": ds.targets.clone()}  # uint8 [N,28,28], int64 [N]
        torch.save(raw, cache)
    x = (raw["x"].float() / 255 - 0.1307) / 0.3081
    tensors = {"x": x[:, None].to(device), "y": raw["y"].to(device)}
    return Dataset(tensors, labels=raw["y"], out_dim=10, metrics=classification_metrics)


def majority(train: bool, device: torch.device | str, size: int = 512, dim: int = 32, seed: int = 0) -> Dataset:
    """Toy task: y = sign(sum(x)) with ~2% label noise (use with the MSE loss)."""
    g = torch.Generator().manual_seed(seed + (0 if train else 1))
    x = torch.randn(size, dim, generator=g)
    y = torch.where(x.sum(1, keepdim=True) > 0, 1.0, -1.0)
    y = torch.where(torch.randn(size, 1, generator=g) > 2, -y, y)
    return Dataset({"x": x.to(device), "y": y.to(device)}, labels=(y[:, 0] > 0).long(), out_dim=1, metrics=sign_metrics)


DATASETS = {"mnist": mnist, "majority": majority}


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
        candidates = (data.labels == self.label) if self.label is not None else torch.ones(len(data), dtype=torch.bool)
        idx = candidates.nonzero()[:, 0]
        perm = torch.randperm(len(idx), generator=torch.Generator().manual_seed(self.seed))
        mask = torch.zeros(len(data), dtype=torch.bool)
        mask[idx[perm[: round(self.frac * len(idx))]]] = True
        return mask
