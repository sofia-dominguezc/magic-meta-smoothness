from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


class Logger:
    """Accumulates per-step tensors on the device and only moves them to the CPU
    (and optionally to wandb) every `flush_every` steps, so training never syncs.
    """

    def __init__(self, wandb_project: str | None = None, name: str = "", config: dict | None = None, flush_every: int = 100):
        self.name = name
        self.flush_every = flush_every
        self.wandb_run = None
        if wandb_project is not None:
            import wandb

            self.wandb_run = wandb.init(project=wandb_project, name=name, config=config)
        self._buf: dict[str, list[tuple[int, Tensor]]] = defaultdict(list)
        self._history: dict[str, list[Tensor]] = defaultdict(list)

    def log(self, step: int, **values: Tensor):
        for k, v in values.items():
            self._buf[k].append((step, v.detach()))
        if sum(map(len, self._buf.values())) >= self.flush_every:
            self.flush()

    def flush(self):
        per_step: dict[int, dict[str, Any]] = defaultdict(dict)
        for k, entries in self._buf.items():
            vals = torch.stack([v for _, v in entries]).cpu()
            self._history[k].append(vals)
            for (s, _), v in zip(entries, vals):
                per_step[s] |= {k: v.item()} if v.ndim == 0 else {f"{k}/{i}": x for i, x in enumerate(v.tolist())}
        self._buf.clear()
        if self.wandb_run is not None:
            for s in sorted(per_step):
                self.wandb_run.log(per_step[s], step=s)

    def finish(self, path: Path):
        """Save the logged metrics to `path` and plot them (one panel per quantity) next to it."""
        self.flush()
        metrics = {k: torch.cat(v).numpy() for k, v in self._history.items()}
        torch.save({k: torch.from_numpy(v) for k, v in metrics.items()}, path)
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5), squeeze=False)
        alpha = 1 - 2 ** (-1 / 10)  # EMA with a half-life of 10 steps
        for ax, (name, arr) in zip(axes[0], metrics.items()):
            if arr.ndim == 1:
                ema = arr.copy()
                for i in range(1, len(ema)):
                    ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
                ax.plot(arr, alpha=0.4, label="raw")
                ax.plot(ema, label="EMA")
            else:  # e.g. eigenvalues: one line per entry
                ax.plot(arr, label=[str(i) for i in range(arr.shape[1])])
            ax.legend(fontsize=7)
            ax.set(title=name, xlabel="step")
        fig.suptitle(self.name)
        fig.tight_layout()
        fig.savefig(path.with_suffix(".png"), dpi=150)
        if self.wandb_run is not None:
            self.wandb_run.finish()


def agreement_metrics(pred: np.ndarray, real: np.ndarray) -> dict[str, float]:
    """How well `pred` matches `real` under the assumption real = pred (slope 1, no offset).
    All of these except `pearson` penalize scale/offset errors, and all are symmetric in (pred, real).

    - pearson:   plain correlation (ignores scale and offset)
    - ccc:       Lin's concordance correlation, 2 cov / (var_p + var_r + (mean_p - mean_r)^2)
    - cosine:    cosine similarity of the raw vectors (slope 1 through the origin, no centering)
    - tls_slope: total-least-squares slope through the origin (1 is perfect; symmetric: slope(r, p) = 1/slope(p, r))
    - rel_rmse:  ||pred - real|| / sqrt(||pred|| ||real||)
    """
    p, r = pred.astype(np.float64), real.astype(np.float64)
    eps = 1e-12
    cov = np.mean((p - p.mean()) * (r - r.mean()))
    d = r @ r - p @ p
    return {
        "pearson": float(cov / (p.std() * r.std() + eps)),
        "ccc": float(2 * cov / (p.var() + r.var() + (p.mean() - r.mean()) ** 2 + eps)),
        "cosine": float((p @ r) / (np.linalg.norm(p) * np.linalg.norm(r) + eps)),
        "tls_slope": float((d + np.sqrt(d**2 + 4 * (p @ r) ** 2)) / (2 * (p @ r) + eps)),
        "rel_rmse": float(np.linalg.norm(p - r) / (np.sqrt(np.linalg.norm(p) * np.linalg.norm(r)) + eps)),
    }


def plot_scatter_grid(
    preds: np.ndarray, reals: np.ndarray, names: list[str], out_path: Path, title: str = ""
) -> list[dict[str, float]]:
    """preds, reals: [n_fns, n_runs] changes of each eval fn (relative to the full model).
    Saves the plot and returns the agreement metrics of each eval fn."""
    n = len(names)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), squeeze=False)
    all_metrics = []
    for i, ax in enumerate(axes.flat):
        if i >= n:
            fig.delaxes(ax)
            continue
        x, y = preds[i], reals[i]
        m = agreement_metrics(x, y)
        all_metrics.append(m)
        lo, hi = min(x.min(), y.min(), 0), max(x.max(), y.max(), 0)
        ax.scatter(x, y, alpha=0.7, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        info = "\n".join(f"{k}: {v:.3f}" for k, v in m.items())
        ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set(xlabel="predicted Δf", ylabel="real Δf", title=names[i])
        ax.title.set_fontsize(9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return all_metrics
