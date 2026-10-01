from collections import defaultdict
from itertools import accumulate
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


class DeviceLogger:
    def __init__(self) -> None:
        self.history: dict[str, list[Tensor]] = defaultdict(list)

    def log(self, **kwargs: Tensor) -> None:
        for k, v in kwargs.items():
            self.history[k].append(v.detach())

    def flush(self) -> dict[str, Tensor]:
        return {
            k: torch.stack(v).cpu() if v and isinstance(v[0], Tensor) else torch.tensor(v)
            for k, v in self.history.items()
        }


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_training_plots(metrics: dict[str, Tensor], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, min(3, len(metrics)), figsize=(12, 3.5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (name, data) in zip(axes, metrics.items()):
        arr = data.numpy() if isinstance(data, Tensor) else np.asarray(data)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        alpha = 1.0 - np.exp(-np.log(2.0) / 10)
        ema = list(accumulate(arr, lambda prev, curr: alpha * curr + (1 - alpha) * prev))
        
        ax.plot(range(len(arr)), arr, alpha=0.4, label="Raw")
        ax.plot(range(len(ema)), ema, label="EMA")
        ax.set(title=name, xlabel="Steps")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def compute_identity_metrics(pred: Tensor, real: Tensor) -> dict[str, float]:
    p = pred.flatten().float()
    r = real.flatten().float()
    cos_sim = torch.nn.functional.cosine_similarity(p, r, dim=0).item()    

    p_centered = p - p.mean()
    r_centered = r - r.mean()
    denom = torch.sqrt(torch.sum(p_centered ** 2) * torch.sum(r_centered ** 2))
    pearson_r = (torch.sum(p_centered * r_centered) / (denom + 1e-8)).item()

    return {
        "cosine_similarity": cos_sim,
        "pearson_r": pearson_r,
    }


def plot_regression_eval(
    all_preds: list[list[float]],
    all_reals: list[list[float]],
    eval_names: list[str],
    out_path: Path,
) -> None:
    n_plots = len(all_preds)
    ncols = min(n_plots, 4)
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx in range(n_plots):
        ax = axes_flat[idx]
        x = np.array(all_preds[idx])
        y = np.array(all_reals[idx])

        ax.scatter(x, y, alpha=0.7, edgecolors="none", label="Models", color="tab:blue")
        
        lim_min = min(x.min(), y.min())
        lim_max = max(x.max(), y.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linestyle="--", linewidth=1.5, label="y = x")

        metrics = compute_identity_metrics(torch.from_numpy(x), torch.from_numpy(y))
        info = f"$R^2_{{id}}$: {metrics['r2_identity']:.3f}\nCos: {metrics['cosine_similarity']:.3f}"
        ax.text(0.05, 0.75, info, transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel("Predicted Output")
        ax.set_ylabel("Real Output")
        ax.set_title(eval_names[idx], fontsize=10)
        ax.legend(loc="lower right", fontsize=8)

    for idx in range(n_plots, len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
