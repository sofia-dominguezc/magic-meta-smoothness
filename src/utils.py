from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def dict_zip[T](*dicts: dict[str, T]) -> dict[str, tuple[T, ...]]:
    """e.g. {"a": 1, "b": 2}, {"a": 3, "b": 4} -> {"a": (1, 3), "b": (2, 4)}"""
    return {key: tuple(d[key] for d in dicts) for key in dicts[0]}


class Logger:
    """Accumulates per-step tensors on the device and only moves them to the CPU
    (and optionally to wandb) every `flush_every` steps, so training never syncs.
    """

    def __init__(
        self,
        wandb_project: str | None = None,
        name: str = "Training run",
        config: dict | None = None,
        flush_every: int = 100,
    ):
        self.name = name
        self.flush_every = flush_every
        self.wandb_run = None
        if wandb_project is not None:
            import wandb

            self.wandb_run = wandb.init(project=wandb_project, name=name, config=config)
        self._buf: dict[str, list[tuple[int, Tensor]]] = defaultdict(list)
        self._history: dict[str, list[tuple[list[int], Tensor]]] = defaultdict(list)  # (steps, values) per flush

    def log(self, step: int, **values: Tensor):
        for k, v in values.items():
            self._buf[k].append((step, v.detach()))
        if sum(map(len, self._buf.values())) >= self.flush_every:
            self.flush()

    def flush(self):
        per_step: dict[int, dict[str, Any]] = defaultdict(dict)
        for k, entries in self._buf.items():
            vals = torch.stack([v for _, v in entries]).cpu()
            self._history[k].append(([s for s, _ in entries], vals))
            for (s, _), v in zip(entries, vals):
                per_step[s] |= {k: v.item()} if v.ndim == 0 else {f"{k}/{i}": x for i, x in enumerate(v.tolist())}
        self._buf.clear()
        if self.wandb_run is not None:
            for s in sorted(per_step):
                self.wandb_run.log(per_step[s], step=s)

    def finish(self, path: Path):
        """Save the logged metrics to `path` as {name: {"step": [n], "value": [n, ...]}} and plot
        them (one panel per quantity) next to it."""
        self.flush()
        metrics = {
            k: {"step": torch.tensor([s for steps, _ in v for s in steps]), "value": torch.cat([vals for _, vals in v])}
            for k, v in self._history.items()
        }
        torch.save(metrics, path)
        # one row per kind: train metrics, eval metrics, then vector-valued ones (e.g. eigenvalues)
        order = lambda k: (metrics[k]["value"].ndim > 1, k.startswith("eval_"))
        names = sorted(metrics, key=order)
        ncols = min(len(names), 3)
        nrows = -(-len(names) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
        for ax in axes.flat[len(names) :]:
            fig.delaxes(ax)
        alpha = 1 - 2 ** (-1 / 10)  # EMA with a half-life of 10 steps
        for ax, name in zip(axes.flat, names):
            m = metrics[name]
            steps, arr = m["step"].numpy(), m["value"].numpy()
            if arr.ndim > 1:  # e.g. eigenvalues: one line per entry
                ax.plot(steps, arr, label=[str(i) for i in range(arr.shape[1])])
            elif np.all(np.diff(steps) == 1):  # logged every step: smooth it
                ema = arr.copy()
                for i in range(1, len(ema)):
                    ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
                ax.plot(steps, arr, alpha=0.4, label="raw")
                ax.plot(steps, ema, label="EMA")
            else:  # e.g. evaluations every few steps
                ax.plot(steps, arr, "o-", label="raw")
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

    - ccc:       Lin's concordance correlation, 2 cov / (var_p + var_r + (mean_p - mean_r)^2)
    - cosine:    cosine similarity of the raw vectors (slope 1 through the origin, no centering)
    - tls_slope: total-least-squares slope through the origin (1 is perfect; symmetric: slope(r, p) = 1/slope(p, r))
    - rel_rmse:  ||pred - real|| / sqrt(||pred|| ||real||)
    """
    p, r = pred.astype(np.float64), real.astype(np.float64)
    eps = 1e-12
    cov = np.mean((p - p.mean()) * (r - r.mean()))
    d = p @ p - r @ r
    tls_slope = (d + np.sqrt(d**2 + 4 * (r @ p) ** 2)) / (2 * (r @ p) + eps)
    return {
        "cosine": float((p @ r) / (np.linalg.norm(p) * np.linalg.norm(r) + eps)),
        "ccc": float(2 * cov / (p.var() + r.var() + (p.mean() - r.mean()) ** 2 + eps)),
        "tls_slope": float(min(tls_slope, 1 / tls_slope)),
        "rel_rmse": float(np.linalg.norm(p - r) / (np.sqrt(np.linalg.norm(p) * np.linalg.norm(r)) + eps)),
    }


SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]  # second encoding: past 3 colors, hues alone aren't distinguishable


def plot_scatter_grid(
    preds: np.ndarray,
    reals: np.ndarray,
    full: np.ndarray,
    positions: list[tuple[int, int]],
    titles: list[str],
    groups: list[str],
    out_path: Path,
    title: str = "",
) -> list[dict[str, float]]:
    """One panel per eval fn i, at grid cell positions[i] = (row, col), with one point per run.
    preds, reals: [n_fns, n_runs] predicted and real values of each eval fn on each run;
    full: [n_fns] its value on the full model; groups: [n_runs] what colors each point.
    The agreement metrics compare the changes (pred - full, real - full); returns them per eval fn."""
    nrows, ncols = max(r for r, _ in positions) + 1, max(c for _, c in positions) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), squeeze=False)
    used = set(positions)
    for (r, c), ax in np.ndenumerate(axes):
        if (r, c) not in used:
            fig.delaxes(ax)
    categories = list(dict.fromkeys(groups))
    group_idx = np.array([categories.index(g) for g in groups])
    all_metrics = []
    for i, (r, c) in enumerate(positions):
        ax, x, y, f = axes[r, c], preds[i], reals[i], full[i]
        m = agreement_metrics(x - f, y - f)
        all_metrics.append(m)
        lo, hi = min(x.min(), y.min(), f), max(x.max(), y.max(), f)
        pad = 0.05 * (hi - lo + 1e-12)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="#52514e", lw=1, ls="--", zorder=1)
        ax.axhline(f, color="#a3a29b", lw=0.8, zorder=0)
        ax.axvline(f, color="#a3a29b", lw=0.8, zorder=0)
        for k, cat in enumerate(categories):
            sel = group_idx == k
            ax.scatter(
                x[sel], y[sel], s=30, color=SERIES_COLORS[k % len(SERIES_COLORS)],
                marker=SERIES_MARKERS[k % len(SERIES_MARKERS)], edgecolors="white", linewidths=0.6,
                label=cat, zorder=2,
            )
        info = f"full-data loss: {f:.4f}\n" + "\n".join(f"{k}: {v:.3f}" for k, v in m.items())
        ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=8, va="top", color="#0b0b0b",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#d6d5cf", alpha=0.9))
        ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
        ax.set_title(titles[i], fontsize=10)
        ax.set_xlabel("predicted loss (MAGIC)", fontsize=9)
        ax.set_ylabel("real loss (retrained)", fontsize=9)
        ax.tick_params(labelsize=8, colors="#52514e")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[positions[0]].get_legend_handles_labels()
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(handles, labels, title="train data dropped from", loc="upper center", ncols=min(len(labels), 6),
               bbox_to_anchor=(0.5, 0.955), fontsize=8, title_fontsize=8, frameon=False)
    fig.savefig(out_path, dpi=150)
    return all_metrics
