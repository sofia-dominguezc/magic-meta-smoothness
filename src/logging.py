import math
import textwrap
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
    Mv = M @ F.normalize(M.T @ u, dim=0, eps=eps)
    for _ in range(n_iter - 1):
        u = F.normalize(Mv, dim=0, eps=eps)
        Mv = M @ F.normalize(M.T @ u, dim=0, eps=eps)
    return Mv.norm(), F.normalize(Mv, dim=0, eps=eps)


def scaled_l2(t: Tensor) -> Tensor:
    """The L2 norm of a parameter-shaped tensor, scaled to be comparable with `op_norm`: vectors: the L2 norm; [out, in, ...]
    tensors as [m, n] matrices: ||t||_F (sqrt(m) + sqrt(n)) / sqrt(m n) = the spectral norm of a Gaussian matrix of that Frobenius
    norm (so it equals `op_norm` for a random-like matrix and is smaller for a low-rank one: the ratio measures the effective rank).
    Like `op_norm`, O(1) for the raw tensors of `models.py`."""
    t = t.detach()
    if t.ndim <= 1:
        return t.norm()
    m, n = t.shape[0], t[0].numel()
    return t.norm() * (m**0.5 + n**0.5) / (m * n) ** 0.5


def ema_smooth(steps: np.ndarray, arr: np.ndarray, half_life: float = EMA_HALF_LIFE) -> np.ndarray:
    """EMA of `arr` in step time: the past decays by 2^(-gap / half_life) between consecutive steps"""
    decay = 2.0 ** (-np.diff(steps) / half_life)
    smooth = arr.astype(np.float64)  # (copies; ints would truncate the average)
    for j in range(1, len(smooth)):
        smooth[j] = decay[j - 1] * smooth[j - 1] + (1 - decay[j - 1]) * arr[j]
    return smooth


def total_l2(tensors) -> Tensor:
    """The L2 norm of a dict of tensors as one vector"""
    return torch.stack([t.detach().pow(2).sum() for t in tensors.values()]).sum().sqrt()


def fig_title(fig, title: str, fontsize: float = 10):
    """`fig.suptitle`, wrapped to the figure width (the layouts don't wrap it, so a long title on a small figure gets cropped)"""
    width = int(fig.get_figwidth() * 72 / (0.6 * fontsize))  # characters per line, at ~0.6 em per character
    fig.suptitle("\n".join(textwrap.fill(line, width) for line in title.split("\n")), fontsize=fontsize)


class Logger:
    """Accumulates per-step scalars on the device and only moves them to the CPU every `flush_every` values,
    so training never syncs.
    - metric names are "panel" or "panel/line": `finish` plots one panel per panel name, one line per line name;
      the line "ema" (the metric computed with an EMA of the params) is drawn dotted gray, unsmoothed; the "d_weights" panel is log scale;
      a panel "<direction>_hess" holds the curvature along a direction v: its lines "Hv" (||H v|| / ||v||) and "vHv" (v^T H v / v^T v)
    - `log_norms` logs the `scaled_l2` norm of each tensor (`total_norms`: one value per name / state group, the total L2 norm,
      plotted as a log-scale scalar panel: "<name>/<group>", or "<prefix>/<group>/<i>" for a name "<prefix>/<i>" so that the
      panel is per group with a line per i, e.g. per eval set)
    - a 1-d logged value is one line per entry
    - `lags`: for each `log_norms` quantity (all its tensors as one vector v_t), the relative displacement |v_t - v_s| / |v_t| over the
      logged step s at each lag |t - s| in `lags`, log scale: "displacement/<quantity>" for the default single lag 1, where a
      quantity "<prefix>/<i>" of a state group is "<prefix>/<group>/<i>" (so the eval sets share a panel per group, like the
      total norms), "displacement/<quantity>/lag<k>" (a line per lag) for several. It keeps the last max(lags) steps of each
      quantity in memory (on the device); `lags=()` logs no displacements
    - the "eig" panel is for the Hessian eigenvalues (top "eig/1", ...; most negative "eig/-1", ...): its lines
      "threshold" (the logged stability threshold) is styled; the constant `eos` is drawn, if set
    """

    def __init__(self, name: str = "logging run", *, flush_every: int = 100, smooth: bool = True, split_norms: bool = False,
                 total_norms: bool = False, lags: tuple[int, ...] = (1,), step_scale: int = 1):
        self.name = name
        self.step_scale = step_scale  # plots show step / step_scale (edge / central flow: GD steps)
        self.flush_every = flush_every
        self.smooth = smooth  # EMA-smooth the scalar plots (when there are enough points)
        self.split_norms = split_norms  # one norms plot per OptimState group (the last component of the panel names)
        self.total_norms = total_norms  # `log_norms`: the total L2 norm instead of one norm per tensor
        self.lags = lags
        self._lag_buf: dict[str, list[tuple[int, Tensor]]] = defaultdict(list)  # quantity -> (step, flat vector) of the last max(lags) steps
        self.eos: float | None = None  # `OptimState.stability_threshold`, drawn in the "eig" panel
        self.log_panels: set[str] = set()  # scalar panels drawn in log scale (besides the norms)
        self.yscale: str | None = None  # forces this scale on the norms and the norm-like scalar panels and "eig" (None: log, "eig" linear)
        self._buf: dict[str, list[tuple[int, Tensor]]] = defaultdict(list)
        self._history: dict[str, list[tuple[list[int], Tensor]]] = defaultdict(list)  # (steps, values) per flush
        self._norm_panels: set[str] = set()

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
                self._norm_panels.add(key.rsplit("/", 1)[0] if "/" in key else key)
                self._buf[key].append((step, total_l2(group_tensors)))
            else:
                self._norm_panels.add(panel)
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
        """Save the logged metrics to `path` (see `metrics`) and plot them next to it:
        - <path>.png: the scalars, one panel per panel name (EMA-smoothed in step time when there are enough points)
        - <path>_norms.png: the norms, one panel per `log_norms` name, one line per tensor
          (if `split_norms`: <path>_norms_<group>.png, one per OptimState group, the panels in a square)
          With `total_norms`, the norms are scalar panels (log scale) of the first plot
        """
        metrics = self.metrics()
        torch.save(metrics, path)
        panels: dict[str, list[str]] = defaultdict(list)  # panel -> metric names
        for k in metrics:
            panels[k.rsplit("/", 1)[0] if "/" in k else k].append(k)
        scalars = {p: names for p, names in panels.items() if p not in self._norm_panels or self.total_norms}
        norms = {p: names for p, names in panels.items() if p in self._norm_panels and not self.total_norms}
        if scalars:
            self._plot_scalars(metrics, scalars, path.with_suffix(".png"))
        if norms:
            self._plot_norms(metrics, norms, path.with_name(f"{path.stem}_norms.png"), split=self.split_norms)

    def _grid(self, n_panels: int, ncols: int | None, size: tuple[float, float]):
        """`ncols` None: as square as possible"""
        ncols = min(ncols or math.ceil(math.sqrt(n_panels)), n_panels)
        nrows = -(-n_panels // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False, layout="constrained")
        for ax in axes.flat[n_panels:]:
            fig.delaxes(ax)
        fig_title(fig, self.name)
        return fig, list(axes.flat[:n_panels])

    EIG_STATS = ("sharpness", "min_eig", "eig_iters", "newton_iters", "residual", "n_critical", "oscillation")  # a second-order step's stats, eig-related

    def _panel_order(self, panel: str) -> int:
        """Train metrics, then test metrics, then the eigenvalues and their stats, then the curvatures / |cos| with the eigenvectors"""
        if panel.endswith(("_eig", "_hess")):
            return 3
        if panel.startswith(("eig", "lobpcg_iters")) or panel in self.EIG_STATS:
            return 2
        return 1 if panel.startswith("test_") else 0

    def _plot_scalars(self, metrics: dict, panels: dict[str, list[str]], out_path: Path):
        panels = dict(sorted(panels.items(), key=lambda item: self._panel_order(item[0])))
        fig, axes = self._grid(len(panels), ncols=4, size=(4.5, 3.4))
        for ax, (panel, names) in zip(axes, panels.items()):
            i = 0  # color index of the next line
            for name in names:
                steps, arr = metrics[name]["step"].numpy() / self.step_scale, metrics[name]["value"].numpy()
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
                elif self.smooth and len(steps) >= 20 and panel != "lr" and color != "0.45":  # enough points: smooth (not the schedule, the EMA of params or the threshold)
                    ax.plot(steps, arr, style, alpha=0.3, lw=0.8, ms=2, color=color)
                    ax.plot(steps, ema_smooth(steps, arr), ls, lw=1.2, color=color, label=label)
                else:
                    ax.plot(steps, arr, style, ms=3, lw=1, color=color, label=label)
            if panel == "eig" and self.eos is not None:
                ax.axhline(self.eos, color="gray", ls=":", lw=1, label=f"stability threshold {self.eos:g}")
            if i > 1 or len(names) > 1 or panel == "eig":
                ax.legend(fontsize=7)
            norm_like = panel == "d_weights" or panel in self._norm_panels or panel in self.log_panels or panel.startswith("displacement/")
            ax.set(title=panel, xlabel="step", yscale=self.yscale if self.yscale and (norm_like or panel == "eig") else "log" if norm_like else "linear")
            if panel.endswith("_eig"):  # |cos|
                ax.set_ylim(0, 1)
            ax.title.set_fontsize(9)
        fig.savefig(out_path, dpi=150)

    def _plot_norms(self, metrics: dict, panels: dict[str, list[str]], out_path: Path, split: bool = False):
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
            fig, axes = self._grid(len(plot_panels), ncols=ncols, size=(6, 3.6))
            for ax, (panel, names) in zip(axes, plot_panels.items()):
                for name in names:
                    tensor = name.rsplit("/", 1)[1]
                    layer, _, kind = tensor.rpartition(".")
                    steps = metrics[name]["step"].numpy() / self.step_scale
                    ax.plot(steps, metrics[name]["value"].numpy(), styles.get(kind, ":"), marker="o" if len(steps) == 1 else None,
                            lw=1, color=colors[layer or kind], label=tensor)
                ax.set(title=panel, xlabel="step", yscale=self.yscale or "log")
                ax.title.set_fontsize(9)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside right center", fontsize=6, frameon=False)
            fig.savefig(plot_path, dpi=150)
