"""`Logger`: records per-step scalars and norms ({name: {"step", "value"}}), and their plots (`plot_metrics`, a function of
the recorded dict alone)."""

import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Collection

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

EMA_HALF_LIFE = 10  # steps, for the smoothing of the scalar plots (whatever the logging interval)
NORM_LIKE = ("d_state", "d_weights", "tangent", "params", "grads", "eigvec", "displacement")  # scalar panels (by first name component) in log scale
EIG_STATS = ("sharpness", "min_eig", "eig_iters", "newton_iters", "residual", "n_critical", "oscillation")  # a second-order step's stats, eig-related


def ema_smooth(steps: np.ndarray, arr: np.ndarray, half_life: float = EMA_HALF_LIFE) -> np.ndarray:
    """EMA of `arr` in step time: the past decays by 2^(-gap / half_life) between consecutive steps"""
    decay = 2.0 ** (-np.diff(steps) / half_life)
    smooth = arr.astype(np.float64)  # (copies; ints would truncate the average)
    for j in range(1, len(smooth)):
        smooth[j] = decay[j - 1] * smooth[j - 1] + (1 - decay[j - 1]) * arr[j]
    return smooth


def fig_title(fig, title: str, fontsize: float = 10):
    """`fig.suptitle`, wrapped to the figure width (the layouts don't wrap it, so a long title on a small figure gets cropped)"""
    width = int(fig.get_figwidth() * 72 / (0.6 * fontsize))  # characters per line, at ~0.6 em per character
    fig.suptitle("\n".join(textwrap.fill(line, width) for line in title.split("\n")), fontsize=fontsize)


def grid(title: str, n_panels: int, ncols: int | None, size: tuple[float, float]):
    """A figure of `n_panels` axes of `size` each (`ncols` None: as square as possible), with its title"""
    ncols = min(ncols or math.ceil(math.sqrt(n_panels)), n_panels)
    nrows = -(-n_panels // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False, layout="constrained")
    for ax in axes.flat[n_panels:]:
        fig.delaxes(ax)
    fig_title(fig, title)
    return fig, list(axes.flat[:n_panels])


def split_panels(metrics: dict) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """The metric names by panel (the name without its last component): (scalar panels, norm panels = those whose lines are tensor names)"""
    panels: dict[str, list[str]] = defaultdict(list)
    for name in metrics:
        panels[name.rsplit("/", 1)[0] if "/" in name else name].append(name)
    norms = {p: names for p, names in panels.items() if any("." in name.rsplit("/", 1)[-1] for name in names)}
    return {p: names for p, names in panels.items() if p not in norms}, norms


def plot_metrics(
    metrics: dict, path: Path, title: str, *, smooth: bool = True, split_norms: bool = False, step_scale: int = 1,
    eos: float | None = None, yscale: str | None = None, log_panels: Collection[str] = (),
):
    """Plot the metrics saved at `path` next to it, one panel per panel name ("panel" or "panel/line"), one line per line name:
    - <path>.png: the scalars. The line "ema" (the metric computed with an EMA of the params) is drawn dotted gray, unsmoothed;
      a panel "<direction>_hess" holds the curvature along a direction v: its lines "Hv" (||H v|| / ||v||) and "vHv" (v^T H v / v^T v);
      the "eig" panel is for the Hessian eigenvalues (top "eig/1", ...; most negative "eig/-1", ...): its line "threshold"
      (the logged stability threshold) is styled, and the constant `eos` is drawn if set; the `NORM_LIKE` panels and
      `log_panels` are in log scale
    - <path>_norms.png: the norms per tensor (the panels whose lines are tensor names), one line per tensor
      (`split_norms`: <path>_norms_<group>.png, one per OptimState group = last component of the panel names, the panels in a square)
    `smooth`: EMA-smooth the scalars in step time (when there are enough points); `step_scale`: the plots show step / step_scale
    (edge / central flow: GD steps); `yscale`: forces this scale on the norms, the norm-like scalar panels and "eig"
    """
    scalars, norms = split_panels(metrics)
    if scalars:
        _plot_scalars(metrics, scalars, path.with_suffix(".png"), title, smooth, step_scale, eos, yscale, log_panels)
    if norms:
        _plot_norms(metrics, norms, path.with_name(f"{path.stem}_norms.png"), title, split_norms, step_scale, yscale)


def _panel_order(panel: str) -> int:
    """Train metrics, then test metrics, then the eigenvalues and their stats, then the curvatures / |cos| with the eigenvectors"""
    if panel.endswith(("_eig", "_hess")):
        return 3
    if panel.startswith(("eig", "lobpcg_iters")) or panel in EIG_STATS:
        return 2
    return 1 if panel.startswith("test_") else 0


def _plot_scalars(metrics: dict, panels: dict[str, list[str]], out_path: Path, title: str, smooth: bool, step_scale: int,
                  eos: float | None, yscale: str | None, log_panels: Collection[str]):
    panels = dict(sorted(panels.items(), key=lambda item: _panel_order(item[0])))
    fig, axes = grid(title, len(panels), ncols=4, size=(4.5, 3.4))
    for ax, (panel, names) in zip(axes, panels.items()):
        i = 0  # color index of the next line
        for name in names:
            steps, arr = metrics[name]["step"].numpy() / step_scale, metrics[name]["value"].numpy()
            order = np.argsort(steps, kind="stable")  # (MAGIC logs the steps in reverse)
            steps, arr = steps[order], arr[order]
            label = name.rsplit("/", 1)[1] if "/" in name else None
            if label == "ema":  # computed with the EMA of the params: dotted gray
                color, ls, label = "0.45", ":", "ema of params"
            elif panel == "eig" and label == "threshold":
                color, ls, label = "0.45", "--", "stability threshold"
            elif panel.endswith("_hess"):
                label = {"Hv": "||Hv|| / ||v||", "vHv": "v^T H v / v^T v"}.get(label, label)
                color, ls, i = f"C{i}", "-", i + 1
            else:
                color, ls, i = f"C{i}", "-", i + 1
            style = f"o{ls}" if len(steps) == 1 or len(steps) > 1 and steps[1] - steps[0] > 1 else ls  # not per-step: show the points
            if arr.ndim == 2:  # a vector per step: one line per entry
                for j in range(arr.shape[1]):
                    ax.plot(steps, arr[:, j], style, ms=3, lw=1, color=f"C{i}", label=f"{label} #{j}" if label else str(j))
                    i += 1
            elif smooth and len(steps) >= 20 and panel != "lr" and color != "0.45":  # enough points: smooth (not the schedule, the EMA of params or the threshold)
                ax.plot(steps, arr, style, alpha=0.3, lw=0.8, ms=2, color=color)
                ax.plot(steps, ema_smooth(steps, arr), ls, lw=1.2, color=color, label=label)
            else:
                ax.plot(steps, arr, style, ms=3, lw=1, color=color, label=label)
        if panel == "eig" and eos is not None:
            ax.axhline(eos, color="gray", ls=":", lw=1, label=f"stability threshold {eos:g}")
        if i > 1 or len(names) > 1 or panel == "eig":
            ax.legend(fontsize=7)
        norm_like = panel.split("/")[0] in NORM_LIKE or panel in log_panels
        ax.set(title=panel, xlabel="step", yscale=yscale if yscale and (norm_like or panel == "eig") else "log" if norm_like else "linear")
        if panel.endswith("_eig"):  # |cos|
            ax.set_ylim(0, 1)
        ax.title.set_fontsize(9)
    fig.savefig(out_path, dpi=150)


def _plot_norms(metrics: dict, panels: dict[str, list[str]], out_path: Path, title: str, split: bool, step_scale: int, yscale: str | None):
    """Lines colored by layer (the tensor name without its last component) and styled by that component.
    `split`: one plot <out_path>_<group>.png per group = last component of the panel names, the panels in a square."""
    tensors = list(dict.fromkeys(n.rsplit("/", 1)[1] for names in panels.values() for n in names))
    layers = list(dict.fromkeys(t.rpartition(".")[0] or t for t in tensors))
    colors = dict(zip(layers, plt.cm.viridis(np.linspace(0, 0.9, len(layers)))))
    styles = {"weight": "-", "bias": "--"}
    if split:
        groups: dict[str, dict[str, list[str]]] = defaultdict(dict)
        for panel, names in panels.items():
            groups[panel.rsplit("/", 1)[-1]][panel] = names
        plots = [(group_panels, out_path.with_name(f"{out_path.stem}_{group}.png"), None) for group, group_panels in groups.items()]
    else:
        plots = [(panels, out_path, 2)]
    for plot_panels, plot_path, ncols in plots:
        fig, axes = grid(title, len(plot_panels), ncols=ncols, size=(6, 3.6))
        for ax, (panel, names) in zip(axes, plot_panels.items()):
            for name in names:
                tensor = name.rsplit("/", 1)[1]
                layer, _, kind = tensor.rpartition(".")
                steps = metrics[name]["step"].numpy() / step_scale
                ax.plot(steps, metrics[name]["value"].numpy(), styles.get(kind, ":"), marker="o" if len(steps) == 1 else None,
                        lw=1, color=colors[layer or kind], label=tensor)
            ax.set(title=panel, xlabel="step", yscale=yscale or "log")
            ax.title.set_fontsize(9)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside right center", fontsize=6, frameon=False)
        fig.savefig(plot_path, dpi=150)


def scaled_l2(t: Tensor) -> Tensor:
    """The L2 norm of a parameter-shaped tensor, scaled like its operator norm: vectors: the L2 norm; [out, in, ...]
    tensors as [m, n] matrices: ||t||_F (sqrt(m) + sqrt(n)) / sqrt(m n) = the spectral norm of a Gaussian matrix of that Frobenius
    norm (smaller than the spectral norm for a low-rank matrix). O(1) for the raw tensors of `models.py`."""
    t = t.detach()
    if t.ndim <= 1:
        return t.norm()
    m, n = t.shape[0], t[0].numel()
    return t.norm() * (m**0.5 + n**0.5) / (m * n) ** 0.5


def total_l2(tensors) -> Tensor:
    """The L2 norm of a dict of tensors as one vector"""
    return torch.stack([t.detach().pow(2).sum() for t in tensors.values()]).sum().sqrt()


class Logger:
    """Accumulates per-step scalars on the device and only moves them to the CPU every `flush_every` values,
    so training never syncs.
    - metric names are "panel" or "panel/line" (see `plot_metrics`); a 1-d logged value is one line per entry
    - `log_norms` logs the `scaled_l2` norm of each tensor (`total_norms`: one value per name / state group, the total L2 norm:
      "<name>/<group>", or "<prefix>/<group>/<i>" for a name "<prefix>/<i>" so that the panel is per group with a line per i,
      e.g. per eval set)
    - `lags`: for each `log_norms` quantity (all its tensors as one vector v_t), the relative displacement |v_t - v_s| / |v_t| over the
      logged step s at each lag |t - s| in `lags`: "displacement/<quantity>" for the default single lag 1, where a
      quantity "<prefix>/<i>" of a state group is "<prefix>/<group>/<i>" (so the eval sets share a panel per group, like the
      total norms), "displacement/<quantity>/lag<k>" (a line per lag) for several. It keeps the last max(lags) steps of each
      quantity in memory (on the device); `lags=()` logs no displacements
    - `plot_options`: of `plot_metrics`, for `finish`
    """

    def __init__(self, name: str = "logging run", *, flush_every: int = 100, total_norms: bool = False, lags: tuple[int, ...] = (1,), **plot_options):
        self.name = name
        self.flush_every = flush_every
        self.total_norms = total_norms  # `log_norms`: the total L2 norm instead of one norm per tensor
        self.lags = lags
        self.plot_options = plot_options
        self._lag_buf: dict[str, list[tuple[int, Tensor]]] = defaultdict(list)  # quantity -> (step, flat vector) of the last max(lags) steps
        self._buf: dict[str, list[tuple[int, Tensor]]] = defaultdict(list)
        self._history: dict[str, list[tuple[list[int], Tensor]]] = defaultdict(list)  # (steps, values) per flush

    def log(self, step: int, **values: Tensor):
        """Log 0-d tensors, or 1-d (e.g. one value per vmapped run: plotted as one line each)."""
        for k, v in values.items():
            assert v.ndim <= 1, f"{k} is not a scalar or vector: {tuple(v.shape)}"
            self._buf[k].append((step, v.detach()))
        self._maybe_flush()

    def log_norms(self, step: int, name: str, tensors):
        """Log the `scaled_l2` norm of each tensor of a dict {tensor name: tensor} as "name/<tensor name>",
        or of an OptimState as "name/<group>/<tensor name>"."""
        groups = {"": tensors} if isinstance(tensors, dict) else {g: getattr(tensors, g) for g in tensors.groups}
        for group, group_tensors in groups.items():
            panel = f"{name}/{group}" if group else name
            prefix, _, index = name.rpartition("/")
            key = f"{prefix}/{group}/{index}" if group and prefix else panel  # the quantities "<prefix>/<i>" share a panel
            if self.total_norms:
                self._buf[key].append((step, total_l2(group_tensors)))
            else:
                for k, t in group_tensors.items():
                    self._buf[f"{panel}/{k}"].append((step, scaled_l2(t)))
            if self.lags:
                self._log_displacement(step, key, group_tensors)
        self._maybe_flush()

    def _log_displacement(self, step: int, key: str, tensors: dict[str, Tensor]):
        """"displacement/<key>[/lag<k>]" = |v_step - v_s| / |v_step| for each buffered step s at a lag |step - s| in `lags`"""
        v = torch.cat([t.detach().flatten() for t in tensors.values()])
        history = self._lag_buf[key]
        for s, u in history:
            if abs(step - s) in self.lags:
                name = f"displacement/{key}" + (f"/lag{abs(step - s)}" if len(self.lags) > 1 else "")
                self._buf[name].append((step, (v - u).norm() / v.norm()))
        history.append((step, v))
        self._lag_buf[key] = [(s, u) for s, u in history if abs(step - s) < max(self.lags)]

    def _maybe_flush(self):
        if sum(map(len, self._buf.values())) >= self.flush_every:
            self.flush()

    def flush(self):
        for k, entries in self._buf.items():
            self._history[k].append(([s for s, _ in entries], torch.stack([v for _, v in entries]).cpu()))
        self._buf.clear()

    def metrics(self) -> dict[str, dict[str, Tensor]]:
        """All the logged metrics so far, as {name: {"step": [n], "value": [n]}}"""
        self.flush()
        return {
            k: {"step": torch.tensor([s for steps, _ in v for s in steps]), "value": torch.cat([vals for _, vals in v])}
            for k, v in self._history.items()
        }

    def restore(self, metrics: dict[str, dict[str, Tensor]]):
        """Start from the saved `metrics()` of a killed run"""
        self._history = defaultdict(list, {k: [(m["step"].tolist(), m["value"])] for k, m in metrics.items()})

    def finish(self, path: Path):
        """Save the logged metrics to `path` (see `metrics`) and plot them next to it (`plot_metrics`)"""
        metrics = self.metrics()
        torch.save(metrics, path)
        plot_metrics(metrics, path, self.name, **self.plot_options)
