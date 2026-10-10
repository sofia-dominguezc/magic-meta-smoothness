from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def dict_zip[T](*dicts: dict[str, T]) -> dict[str, tuple[T, ...]]:
    """e.g. {"a": 1, "b": 2}, {"a": 3, "b": 4} -> {"a": (1, 3), "b": (2, 4)}"""
    return {key: tuple(d[key] for d in dicts) for key in dicts[0]}


def op_norm(t: Tensor, u: Tensor | None = None, n_iter: int = 3) -> tuple[Tensor, Tensor | None]:
    """Operator norm of a parameter-shaped tensor, without syncs:
    - vectors (ndim <= 1): the L2 norm
    - [out, in, ...] tensors, as [out, -1] matrices: the spectral norm, by `n_iter` power iterations
      (2 matrix-vector products each) from `u` ([out]; random if None)
    Returns: (norm, u), with `u` to warm-start the next call. The norm is a lower bound that converges over calls
    (for a Gaussian matrix, within ~10% after the first call).

    In the spectral μP of `models.py` this is the norm in which every raw parameter and gradient is O(1) in every
    layer (for the effective weight W = sqrt(fan_out / fan_in) w it is the RMS-to-RMS operator norm, i.e. how much
    the layer's output RMS changes per unit input RMS), and a step changes each parameter by O(lr)."""
    t = t.detach()
    if t.ndim <= 1:
        return t.norm(), None
    M = t.flatten(1)
    if u is None:
        u = torch.randn(M.shape[0], generator=torch.Generator().manual_seed(0)).to(M)
    for _ in range(n_iter):
        v = F.normalize(M.T @ u, dim=0)
        Mv = M @ v
        u = F.normalize(Mv, dim=0)
    return Mv.norm(), u


class Logger:
    """Accumulates per-step scalars on the device and only moves them to the CPU (and optionally to wandb)
    every `flush_every` values, so training never syncs.
    - metric names are "panel" or "panel/line": `finish` plots one panel per panel name, one line per line name
    - `log_norms` logs the `op_norm`, with `init_power_iters` power iterations the first time a tensor is seen
      and `power_iters` (warm-started) afterwards
    - the "eig" panel is for the normalized top Hessian eigenvalues: the edge of stability is drawn at 2
    """

    def __init__(
        self,
        wandb_project: str | None = None,
        name: str = "logging run",
        config: dict | None = None,
        *,
        flush_every: int = 100,
        init_power_iters: int = 10,
        power_iters: int = 3,
    ):
        self.name = name
        self.flush_every = flush_every
        self.init_power_iters = init_power_iters
        self.power_iters = power_iters
        self.wandb_run = None
        if wandb_project is not None:
            import wandb

            self.wandb_run = wandb.init(project=wandb_project, name=name, config=config)
            self.wandb_run.define_metric("step")
            self.wandb_run.define_metric("*", step_metric="step")  # steps need not increase (backward pass)
        self._buf: dict[str, list[tuple[int, Tensor]]] = defaultdict(list)
        self._history: dict[str, list[tuple[list[int], Tensor]]] = defaultdict(list)  # (steps, values) per flush
        self._u: dict[str, Tensor | None] = {}  # power iteration vectors, by metric name
        self._norm_panels: set[str] = set()

    def log(self, step: int, **values: Tensor):
        """Log 0-d tensors."""
        for k, v in values.items():
            assert v.ndim == 0, f"{k} is not a scalar: {tuple(v.shape)}"
            self._buf[k].append((step, v.detach()))
        self._maybe_flush()

    def log_norms(self, step: int, name: str, tensors):
        """Log the `op_norm` of each tensor of a dict {tensor name: tensor} as "name/<tensor name>",
        or of an OptimState as "name/<group>/<tensor name>"."""
        groups: dict[str, dict[str, Tensor]] = {"": tensors} if isinstance(tensors, dict) else tensors.param_groups
        for group, group_tensors in groups.items():
            panel = f"{name}/{group}" if group else name
            self._norm_panels.add(panel)
            for k, t in group_tensors.items():
                key = f"{panel}/{k}"
                if (u := self._u.get(key)) is None:
                    u = op_norm(t, None, self.init_power_iters)[1]
                norm, self._u[key] = op_norm(t, u, self.power_iters)
                self._buf[key].append((step, norm))
        self._maybe_flush()

    def _maybe_flush(self):
        if sum(map(len, self._buf.values())) >= self.flush_every:
            self.flush()

    def flush(self):
        per_step: dict[int, dict[str, Any]] = defaultdict(dict)
        for k, entries in self._buf.items():
            vals = torch.stack([v for _, v in entries]).cpu()
            self._history[k].append(([s for s, _ in entries], vals))
            for (s, _), v in zip(entries, vals):
                per_step[s][k] = v.item()
        self._buf.clear()
        if self.wandb_run is not None:
            for s in sorted(per_step):
                self.wandb_run.log({**per_step[s], "step": s})

    def finish(self, path: Path):
        """Save the logged metrics to `path` as {name: {"step": [n], "value": [n]}} and plot them next to it:
        - <path>.png: the scalars, one panel per panel name (smoothed if logged every step)
        - <path>_norms.png: the norms, one panel per `log_norms` name, one line per tensor, log scale
        """
        self.flush()
        metrics = {
            k: {"step": torch.tensor([s for steps, _ in v for s in steps]), "value": torch.cat([vals for _, vals in v])}
            for k, v in self._history.items()
        }
        torch.save(metrics, path)
        panels: dict[str, list[str]] = defaultdict(list)  # panel -> metric names
        for k in metrics:
            panels[k.rsplit("/", 1)[0] if "/" in k else k].append(k)
        scalars = {p: names for p, names in panels.items() if p not in self._norm_panels}
        norms = {p: names for p, names in panels.items() if p in self._norm_panels}
        if scalars:
            self._plot_scalars(metrics, scalars, path.with_suffix(".png"))
        if norms:
            self._plot_norms(metrics, norms, path.with_name(f"{path.stem}_norms.png"))
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _grid(self, n_panels: int, ncols: int, size: tuple[float, float]):
        ncols = min(ncols, n_panels)
        nrows = -(-n_panels // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False, layout="constrained")
        for ax in axes.flat[n_panels:]:
            fig.delaxes(ax)
        fig.suptitle(self.name, fontsize=10)
        return fig, list(axes.flat[:n_panels])

    def _plot_scalars(self, metrics: dict, panels: dict[str, list[str]], out_path: Path):
        fig, axes = self._grid(len(panels), ncols=4, size=(4.5, 3.4))
        alpha = 1 - 2 ** (-1 / 10)  # EMA with a half-life of 10 steps
        for ax, (panel, names) in zip(axes, panels.items()):
            for i, name in enumerate(names):
                steps, arr = metrics[name]["step"].numpy(), metrics[name]["value"].numpy()
                label = name.rsplit("/", 1)[1] if "/" in name else None
                if len(steps) > 50:  # per-step metric: smooth it
                    ema = arr.astype(np.float64)  # (copies; ints would truncate the average)
                    for j in range(1, len(ema)):
                        ema[j] = alpha * arr[j] + (1 - alpha) * ema[j - 1]
                    ax.plot(steps, arr, alpha=0.3, lw=0.8, color=f"C{i}")
                    ax.plot(steps, ema, lw=1.2, color=f"C{i}", label=label)
                else:
                    ax.plot(steps, arr, "o-", ms=3, lw=1, color=f"C{i}", label=label)
            if panel == "eig":
                ax.axhline(2, color="gray", ls="--", lw=0.8, label="edge of stability")
            if len(names) > 1 or panel == "eig":
                ax.legend(fontsize=7)
            ax.set(title=panel, xlabel="step")
            ax.title.set_fontsize(9)
        fig.savefig(out_path, dpi=150)

    def _plot_norms(self, metrics: dict, panels: dict[str, list[str]], out_path: Path):
        """Lines colored by layer (the tensor name without its last component) and styled by that component."""
        tensors = list(dict.fromkeys(n.rsplit("/", 1)[1] for names in panels.values() for n in names))
        layers = list(dict.fromkeys(t.rpartition(".")[0] or t for t in tensors))
        colors = dict(zip(layers, plt.cm.viridis(np.linspace(0, 0.9, len(layers)))))
        styles = {"weight": "-", "bias": "--"}
        fig, axes = self._grid(len(panels), ncols=2, size=(6, 3.6))
        for ax, (panel, names) in zip(axes, panels.items()):
            for name in names:
                tensor = name.rsplit("/", 1)[1]
                layer, _, kind = tensor.rpartition(".")
                ax.plot(metrics[name]["step"].numpy(), metrics[name]["value"].numpy(), styles.get(kind, ":"),
                        lw=1, color=colors[layer or kind], label=tensor)
            ax.set(title=panel, xlabel="step", yscale="log")
            ax.title.set_fontsize(9)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside right center", fontsize=6, frameon=False)
        fig.savefig(out_path, dpi=150)


def agreement_metrics(pred: np.ndarray, real: np.ndarray) -> dict[str, float]:
    """How well `pred` matches `real` under the assumption real = pred (slope 1, no offset).
    All are symmetric in (pred, real) and independent of their number.
    - ccc:       Lin's concordance correlation, 2 cov / (var_p + var_r + (mean_p - mean_r)^2) = 1 - E(p - r)^2 / (...):
                 the R^2 of the identity line; in [-1, 1], 1 iff pred = real; penalizes slope and offset errors
    - spearman:  rank correlation; in [-1, 1]; ignores scale and offset, robust to a few large errors
    - cosine:    cosine similarity of the raw vectors (slope 1 through the origin, no centering)
    - tls_slope: total-least-squares slope through the origin (1 is perfect; symmetric: slope(r, p) = 1/slope(p, r))
    - rel_rmse:  ||pred - real|| / sqrt(||pred|| ||real||)
    """
    p, r = pred.astype(np.float64), real.astype(np.float64)
    eps = 1e-12
    cov = np.mean((p - p.mean()) * (r - r.mean()))
    d = p @ p - r @ r
    tls_slope = (d + np.sqrt(d**2 + 4 * (r @ p) ** 2)) / (2 * (r @ p) + eps)
    ranks = [np.argsort(np.argsort(x)).astype(np.float64) for x in (p, r)]
    return {
        "ccc": float(2 * cov / (p.var() + r.var() + (p.mean() - r.mean()) ** 2 + eps)),
        "spearman": float(np.corrcoef(*ranks)[0, 1]) if len(p) > 1 else float("nan"),
        "cosine": float((p @ r) / (np.linalg.norm(p) * np.linalg.norm(r) + eps)),
        "tls_slope": float(min(tls_slope, 1 / tls_slope)),
        "rel_rmse": float(np.linalg.norm(p - r) / (np.sqrt(np.linalg.norm(p) * np.linalg.norm(r)) + eps)),
    }


def eos_summary(eig: np.ndarray, threshold: float = 2.0, half_life: int = 10) -> dict[str, float | None]:
    """Did a run reach the edge of stability, and for how long? From the normalized top eigenvalues per step:
    - eos_step: first step whose EMA (half-life `half_life` steps) reaches 0.95 * threshold (None: never)
    - eos_frac: fraction of the steps from eos_step on with eig > threshold (0 if never)
    - max_ema:  max of the EMA
    """
    alpha = 1 - 2 ** (-1 / half_life)
    ema = eig.astype(np.float64).copy()
    for i in range(1, len(ema)):
        ema[i] = alpha * eig[i] + (1 - alpha) * ema[i - 1]
    reached = np.nonzero(ema >= 0.95 * threshold)[0]
    first = int(reached[0]) if len(reached) else None
    return {
        "eos_step": first,
        "eos_frac": float((eig[first:] > threshold).mean()) if first is not None else 0.0,
        "max_ema": float(ema.max()),
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
