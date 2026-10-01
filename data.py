from dataclasses import asdict, dataclass
from functools import partial
from typing import Callable
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def get_targets(dataset: Dataset) -> Tensor:
    if hasattr(dataset, "targets"):
        t = dataset.targets
        return t if isinstance(t, Tensor) else torch.tensor(t)
    elif hasattr(dataset, "labels"):
        l = dataset.labels
        return l if isinstance(l, Tensor) else torch.tensor(l)
    return torch.tensor([dataset[i][1] for i in range(len(dataset))])


@dataclass(kw_only=True, frozen=True)
class SubsetSpec:
    frac: float = 0.1
    drop: bool = True
    label: int | None = None
    seed: int = 0

    @property
    def id(self) -> str:
        lbl = "global" if self.label is None else f"lbl{self.label}"
        mode = "drop" if self.drop else "keep"
        return f"{lbl}_{mode}{self.frac:.2f}_s{self.seed}"

    @property
    def rng(self) -> torch.Generator:
        return torch.Generator().manual_seed(self.seed)

    def apply(self, dataset: Dataset) -> tuple[Subset, Tensor]:
        targets = get_targets(dataset)
        subset_mask = (targets == self.label) if self.label is not None else torch.ones_like(targets, dtype=torch.bool)

        keep_idxs = torch.nonzero(~subset_mask).squeeze(-1)
        candidate_idxs = torch.nonzero(subset_mask).squeeze(-1)

        shuffled = candidate_idxs[torch.randperm(len(candidate_idxs), generator=self.rng)]
        keep_frac = (1.0 - self.frac) if self.drop else self.frac
        keep_count = int(len(candidate_idxs) * keep_frac)

        selected_candidates = shuffled[:keep_count]
        dropped_candidates = shuffled[keep_count:]

        final_idxs = torch.cat([keep_idxs, selected_candidates]).sort().values
        return Subset(dataset, final_idxs.tolist()), dropped_candidates

    def to_dict(self) -> dict:
        d = asdict(self)
        d["id"] = self.id
        return d


def get_mnist_dataset(train: bool, root: str = "./data") -> Dataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return datasets.MNIST(root, train=train, download=True, transform=transform)


def _eval_loss_fn(params: dict[str, Tensor], model_arch: torch.nn.Module, X: Tensor, Y: Tensor) -> Tensor:
    logits = torch.func.functional_call(model_arch, params, X)
    return torch.nn.functional.cross_entropy(logits, Y, reduction="sum")


def create_eval_fn(
    test_dataset: Dataset,
    spec: SubsetSpec,
    model_arch: torch.nn.Module,
    device: torch.device | str,
) -> Callable[[dict[str, Tensor]], Tensor]:
    assert not spec.drop
    subset, _ = spec.apply(test_dataset)

    # TODO: eval in batches
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
    X, Y = next(iter(loader))
    X = X.to(device, non_blocking=True)
    Y = Y.to(device, non_blocking=True)

    return partial(_eval_loss_fn, model_arch=model_arch, X=X, Y=Y)
