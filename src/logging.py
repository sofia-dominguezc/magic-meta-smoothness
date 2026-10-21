from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

EMA_HALF_LIFE = 10  # steps, for the smoothing of the scalar plots (whatever the logging interval)


def op_norm(t: Tensor, u: Tensor | None = None, n_iter: int = 3) -> tuple[Tensor, Tensor | None]:
    """Operator norm of a parameter-shaped tensor, without syncs:
    - vectors (ndim <= 1): the L2 norm
    - [out, in, ...] tensors, as [out, -1] matrices: the spectral norm, by `n_iter` power iterations
      (2 matrix-vector products each) from `u` ([out]; random if None)
    Returns: (norm, u), with `u` to warm-start the next call. The norm is a lower bound that converges over calls
    (for a Gaussian matrix, within ~10% after the first call). It is the norm in which the raw tensors of `models.py`
    are O(1)."""
    t = t.detach()
    if t.ndim <= 1:
        return t.norm(), None
    M = t.flatten(1)
    r = torch.randn(M.shape[0], generator=torch.Generator().manual_seed(0)).to(M)
    u = r if u is None else torch.where(u.norm() > 0, u, r)  # a zero `u` (from a zero tensor) would stay zero
    eps = torch.finfo(M.dtype).tiny  # F.normalize's default 1e-12 breaks the iteration on tensors of norm < 1e-12
    for _ in range(n_iter):
        v = F.normalize(M.T @ u, dim=0, eps=eps)
        Mv = M @ v
        u = F.normalize(Mv, dim=0, eps=eps)
    return Mv.norm(), u


class Logger:
    """Accumulates per-step scalars on the device and only moves them to the CPU every `flush_every` values,
    so training never syncs.
    - metric names are "panel" or "panel/line": `finish` plots one panel per panel name, one line per line name
    - `log_norms` logs the `op_norm`, with 10 power iterations the first time a tensor is seen and 3 (warm-started)
      afterwards
    - the "eig" panel is for the Hessian eigenvalues (top "eig/1", ...; most negative "eig/-1", ...; of the
      preconditioned Hessian, so the lr is absorbed): the stability threshold `eos` is drawn, if set
    """

    def __init__(self, name: str = "logging run", *, flush_every: int = 100):
        self.name = name
        self.flush_every = flush_every
        self.eos: float | None = None  # `OptimState.stability_threshold`, drawn in the "eig" panel
        self._buf: dict[str, list[tuple[int, Tensor]]] = defaultdict(list)
        self._history: dict[str, list[tuple[list[int], Tensor]]] = defaultdict(list)  # (steps, values) per flush
        self._running_eigs: dict[str, Tensor | None] = {}
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
        groups = {"": tensors} if isinstance(tensors, dict) else {g: getattr(tensors, g) for g in tensors.groups}
        for group, group_tensors in groups.items():
            panel = f"{name}/{group}" if group else name
            self._norm_panels.add(panel)
            for k, t in group_tensors.items():
                key = f"{panel}/{k}"
                if (u := self._running_eigs.get(key)) is None:
                    u = op_norm(t, None, n_iter=10)[1]
                norm, self._running_eigs[key] = op_norm(t, u, n_iter=3)
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
        - <path>.png: the scalars, one panel per panel name (EMA-smoothed in step time when there are enough points)
        - <path>_norms.png: the norms, one panel per `log_norms` name, one line per tensor
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
        for ax, (panel, names) in zip(axes, panels.items()):
            for i, name in enumerate(names):
                steps, arr = metrics[name]["step"].numpy(), metrics[name]["value"].numpy()
                label = name.rsplit("/", 1)[1] if "/" in name else None
                style = "o-" if len(steps) > 1 and steps[1] - steps[0] > 1 else "-"  # not per-step: show the points
                if len(steps) >= 20 and panel != "lr":  # enough points: smooth (not the schedule)
                    decay = 2.0 ** (-np.diff(steps) / EMA_HALF_LIFE)  # EMA in step time: the past decays by 2^(-gap / half-life)
                    ema = arr.astype(np.float64)  # (copies; ints would truncate the average)
                    for j in range(1, len(ema)):
                        ema[j] = decay[j - 1] * ema[j - 1] + (1 - decay[j - 1]) * arr[j]
                    ax.plot(steps, arr, style, alpha=0.3, lw=0.8, ms=2, color=f"C{i}")
                    ax.plot(steps, ema, lw=1.2, color=f"C{i}", label=label)
                else:
                    ax.plot(steps, arr, style, ms=3, lw=1, color=f"C{i}", label=label)
            if panel == "eig" and self.eos is not None:
                ax.axhline(self.eos, color="gray", ls=":", lw=1, label=f"stability threshold {self.eos:g}")
            if len(names) > 1 or panel == "eig":
                ax.legend(fontsize=7)
            ax.set(title=panel, xlabel="step")
            ax.title.set_fontsize(9)
        fig.savefig(out_path, dpi=150)

    def _plot_norms(self, metrics: dict, panels: dict[str, list[str]], out_path: Path, yscale='log'):
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
            ax.set(title=panel, xlabel="step", **({"yscale": yscale} if yscale is not None else {}))
            ax.title.set_fontsize(9)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside right center", fontsize=6, frameon=False)
        fig.savefig(out_path, dpi=150)
