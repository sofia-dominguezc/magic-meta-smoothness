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
        # by key or group: legend names of the entries of vector metrics (default 1, 2, ...), where
        # entries with the same name share a color; the number of panels per row (default 3); y labels
        self.labels: dict[str, list[str]] = {}
        self.ncols: dict[str, int] = {}
        self.ylabels: dict[str, str] = {}

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

    def finish(self, path: Path, separate: dict[str, str] | None = None):
        """Save the logged metrics to `path` as {name: {"step": [n], "value": [n, ...]}} and plot
        them next to it: one row (or more) per group "<group>/<name>", with the train metrics
        (no group) first, then eval, then the others in logging order.
        With `separate` ({group: title}), each of these groups goes to its own file `<path>_<group>.png`
        instead, titled "<name>: <title>".
        A key "<name>_ids" gives ids for <name>'s entries (e.g. dataset indices), written on the
        lines wherever they change."""
        self.flush()
        metrics = {
            k: {"step": torch.tensor([s for steps, _ in v for s in steps]), "value": torch.cat([vals for _, vals in v])}
            for k, v in self._history.items()
        }
        torch.save(metrics, path)

        groups: dict[str, list[str]] = {"": [], "eval": []}
        for k in metrics:
            if not k.endswith("_ids"):
                groups.setdefault(k.split("/")[0] if "/" in k else "", []).append(k)
        if separate is None:
            figures = [(path.with_suffix(".png"), self.name, list(groups.values()))]
        else:
            figures = [(path.with_name(f"{path.stem}_{g}.png"), f"{self.name}: {t}", [groups[g]]) for g, t in separate.items()]
        for out_path, title, figure_groups in figures:
            self._plot(metrics, figure_groups, title, out_path, strip_group=separate is not None)
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _plot(self, metrics: dict, groups: list[list[str]], title: str, out_path: Path, strip_group: bool):
        group_of = lambda name: name.split("/")[0] if "/" in name else ""
        rows = []
        for names in groups:
            n = self.ncols.get(group_of(names[0]), 3) if names else 3
            rows += [names[i : i + n] for i in range(0, len(names), n)]
        ncols = max(map(len, rows))
        width, height = (6.5, 4.6) if ncols <= 2 else (4.5, 3.4)  # few panels: make them bigger
        fig, axes = plt.subplots(len(rows), ncols, figsize=(width * ncols, height * len(rows)), squeeze=False, layout="constrained")
        alpha = 1 - 2 ** (-1 / 10)  # EMA with a half-life of 10 steps
        for row, names in zip(axes, rows):
            for ax in row[len(names) :]:
                fig.delaxes(ax)
            for ax, name in zip(row, names):
                steps, arr = metrics[name]["step"].numpy(), metrics[name]["value"].numpy()
                if arr.ndim > 1:  # one line per entry, colored by its name
                    default = [str(i + 1) for i in range(arr.shape[1])]
                    labels = self.labels.get(name, self.labels.get(group_of(name), default))
                    colors = {label: f"C{i % 10}" for i, label in enumerate(dict.fromkeys(labels))}
                    ids = metrics[name + "_ids"]["value"].numpy() if name + "_ids" in metrics else None
                    for j, label in enumerate(labels):
                        first = labels.index(label) == j
                        ax.plot(steps, arr[:, j], lw=1, color=colors[label], label=label if first else "_nolegend_")
                        if ids is not None:
                            for r in np.nonzero(np.r_[True, ids[1:, j] != ids[:-1, j]])[0]:
                                ax.annotate(str(ids[r, j]), (steps[r], arr[r, j]), fontsize=6, color=colors[label],
                                            xytext=(1, 1), textcoords="offset points")
                elif "/" not in name:  # train metrics, logged every step: smooth them
                    ema = arr.astype(np.float64)  # (copies; ints would truncate the average)
                    for i in range(1, len(ema)):
                        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
                    ax.plot(steps, arr, alpha=0.4, label="raw")
                    ax.plot(steps, ema, label="EMA")
                else:
                    ax.plot(steps, arr, "o-" if len(steps) < 50 else "-", ms=4)
                if ax.get_legend_handles_labels()[0]:
                    many = len(ax.get_legend_handles_labels()[0]) > 6
                    ax.legend(fontsize=6 if many else 7, **(dict(loc="upper left", bbox_to_anchor=(1.01, 1)) if many else {}))
                ylabel = self.ylabels.get(name, self.ylabels.get(group_of(name), ""))
                ax.set(title=name.split("/", 1)[-1] if strip_group else name, xlabel="step", ylabel=ylabel)
                ax.title.set_fontsize(9)
        fig.suptitle(title, fontsize=10)
        fig.savefig(out_path, dpi=150)


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
