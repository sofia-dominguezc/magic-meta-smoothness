from collections import defaultdict
from pathlib import Path

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
    """Accumulates per-step scalars on the device and only moves them to the CPU every `flush_every` values,
    so training never syncs.
    - metric names are "panel" or "panel/line": `finish` plots one panel per panel name, one line per line name
    - `log_norms` logs the `op_norm`, with 10 power iterations the first time a tensor is seen and 3 (warm-started)
      afterwards
    - the "eig" panel is for the normalized top Hessian eigenvalues: the edge of stability is drawn at 2
    """

    def __init__(self, name: str = "logging run", *, flush_every: int = 100):
        self.name = name
        self.flush_every = flush_every
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
                    u = op_norm(t, None, n_iter=10)[1]
                norm, self._u[key] = op_norm(t, u, n_iter=3)
                self._buf[key].append((step, norm))
        self._maybe_flush()

    def _maybe_flush(self):
        if sum(map(len, self._buf.values())) >= self.flush_every:
            self.flush()

    def flush(self):
        for k, entries in self._buf.items():
            self._history[k].append(([s for s, _ in entries], torch.stack([v for _, v in entries]).cpu()))
        self._buf.clear()

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
