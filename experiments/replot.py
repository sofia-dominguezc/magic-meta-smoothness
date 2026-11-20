"""
Redraw <dir>/metrics.png (scalars) and <dir>/metrics_norms[_<group>].png (norms) from the saved <dir>/metrics.pt of DIR/full_model,
DIR/magic and DIR/rayleigh, without retraining (e.g. after a change of `logging.plot_metrics`, or
to switch the scales):
  python -m experiments.replot all DIR [DIR ...] [--yscale log|linear]
  python -m experiments.replot zoom DIR --start S --n-steps N   # only the steps S to S + N (of the config: edge flow substeps
      are scaled) of DIR/full_model and DIR/magic, into <dir>/metrics_zoom_<S>_<S+N>[_norms...].png next to the full plots
--yscale: of all the norm panels (full_model: params, grads, tangent, eigvec; magic: d_state) and of the norm-like scalar panels
(d_weights, displacements, total norms) and the eigenvalues "eig" (default: the Logger's, log for the norms and linear for "eig").
--log-derivative: also <dir>/metrics_norms_dlog.png, for the "tangent/..." (full_model) and "d_state" (magic, rayleigh) panels: the
growth per step of y_n = max op norm over the panel's tensors (y_{n+1} / y_n for the tangent, y_n / y_{n+1} for df/ds_t;
linear scale) with |1 - P eig| for the eigenvalues "eig/i" logged by `train` (P = stability threshold / its logged "eig/threshold").
"""

import re
from pathlib import Path
from typing import Any

import click
import torch

from main import Experiment, load_renamed
from src.logging import grid, plot_metrics, split_panels


def slice_steps(metrics: dict, steps: tuple[int, int]) -> dict:
    """The metrics {name: {"step", "value"}} restricted to the logged steps in [steps[0], steps[1]] (metrics logged at none dropped)"""
    out = {}
    for k, m in metrics.items():
        keep = (m["step"] >= steps[0]) & (m["step"] <= steps[1])
        if keep.any():
            out[k] = {"step": m["step"][keep], "value": m["value"][keep]}
    return out


def replot(metrics_path: Path, name: str, yscale: str | None = None, eos: float | None = None, log_derivative: bool = False, eigs: dict | None = None,
           smooth: bool = True, split_norms: bool = False, step_scale: int = 1, steps: tuple[int, int] | None = None):
    """`yscale`: forces "log" / "linear" on the norms, the norm-like scalars and "eig" (None: the defaults).
    `eos`: the stability threshold drawn in the "eig" panel (rayleigh: of the preconditioned Hessian). `eigs`: metrics with
    {"eig/i": {"step", "value"}} drawn as |1 - P eig| in the derivative plot, with P = eos / "eig/threshold" (default: those of
    `metrics_path`; without "eig/threshold", |1 - eig|). `split_norms`, `step_scale`: see `plot_metrics`. `steps`: only the logged
    steps (unscaled) in this range, into <stem>_zoom_<start>_<end>*.png (the config's steps) instead of <stem>*.png"""
    metrics = load_renamed(metrics_path)
    all_eigs = metrics if eigs is None else eigs
    if steps is not None:
        metrics, all_eigs = slice_steps(metrics, steps), slice_steps(all_eigs, steps)
        metrics_path = metrics_path.with_stem(f"{metrics_path.stem}_zoom_{steps[0] // step_scale}_{steps[1] // step_scale}")
        name = f"{name}\nsteps {steps[0] // step_scale}-{steps[1] // step_scale}"
    eigs = {k: v for k, v in all_eigs.items() if re.fullmatch(r"eig/-?\d+", k)}
    threshold = all_eigs.get("eig/threshold")
    scalars, norms = split_panels(metrics)
    plot_metrics(metrics, metrics_path, name, smooth=smooth, split_norms=split_norms, step_scale=step_scale, yscale=yscale,
                 eos=eos if "eig/threshold" not in metrics else None)  # (`train` logs its threshold as a line)
    derivs = {p: names for p, names in norms.items() if p.startswith(("tangent", "d_state"))}  # directions propagated by the training
    if derivs and log_derivative:
        fig, axes = grid(name, len(derivs), ncols=2, size=(10, 6))
        for ax, (panel, names) in zip(axes, derivs.items()):
            steps = metrics[names[0]]["step"]
            assert all(torch.equal(metrics[k]["step"], steps) for k in names)
            order = steps.argsort()  # MAGIC logs the steps in reverse
            steps, y = steps[order], torch.stack([metrics[k]["value"][order] for k in names]).max(0).values  # max op norm over the tensors
            backward = panel.startswith("d_state")  # df/ds_t is propagated backward: its growth per step is y_n / y_(n+1)
            ratio, label = (y[:-1] / y[1:], "y_n / y_(n+1)") if backward else (y[1:] / y[:-1], "y_(n+1) / y_n")
            ax.plot(steps[:-1].numpy(), ratio.numpy(), lw=1, color="C0", label=f"{label}, y = max op norm")
            for i, k in enumerate(sorted(eigs, key=lambda k: int(k.split("/")[1]), reverse=True)):
                e_steps, e = eigs[k]["step"], eigs[k]["value"]
                if threshold is not None and eos is not None:  # P = eos / threshold at the eigenvalue's steps
                    thr = threshold["value"][torch.searchsorted(threshold["step"], e_steps).clamp(max=len(threshold["step"]) - 1)]
                    e, label = eos * e / thr, f"|1 - P {k}|"
                else:
                    label = f"|1 - {k}|"
                ax.plot(e_steps.numpy(), (1 - e).abs().numpy(), "o-", ms=3, lw=1, color=f"C{i + 1}", label=label)
            ax.axhline(1, color="gray", lw=0.5)
            ax.set(title=panel, xlabel="step")
            ax.legend(fontsize=7)
        fig.savefig(metrics_path.with_name(f"{metrics_path.stem}_norms_dlog.png"), dpi=150)
    click.echo(f"{metrics_path.parent}: scalars {list(scalars)}, norms {list(norms)}" + (f" ({metrics_path.name})" if steps else ""))


def replot_experiment(exp_dir: Path, yscale: str | None, log_derivative: bool, rayleigh: bool = True, steps: tuple[int, int] | None = None):
    """`replot` DIR/full_model, DIR/magic* and (if `rayleigh`) DIR/rayleigh; `steps` (of the config): only this range (see `replot`)"""
    exp = Experiment(exp_dir, "cpu")
    eos = exp.init_state.stability_threshold
    eigs = torch.load(exp_dir / "full_model" / "metrics.pt", weights_only=True) if (exp_dir / "full_model" / "metrics.pt").exists() else {}
    common = dict(yscale=yscale, log_derivative=log_derivative, step_scale=exp.substeps, steps=None if steps is None else (steps[0] * exp.substeps, steps[1] * exp.substeps))
    plots: list[tuple[str, str, dict[str, Any]]] = [
        ("full_model", exp.name, dict(eos=eos, smooth=exp.cfg.optimizer != "centralflow")),
        *([("magic", f"MAGIC: df/ds_t and df/dw_t; {exp.name}", dict(eigs=eigs, smooth=False, split_norms=True))] if (exp_dir / "magic").exists() else []),
    ]
    if rayleigh:
        plots.append(("rayleigh", f"rayleigh: final Hessian eigenvectors along training, MAGIC's df/ds_t; {exp.name}", dict(eigs=eigs, smooth=False)))
    for sub, name, kwargs in plots:
        if (exp_dir / sub / "metrics.pt").exists():
            replot(exp_dir / sub / "metrics.pt", name, **common, **kwargs)


yscale_option = click.option("--yscale", type=click.Choice(["log", "linear"]), default=None, help="of all the norm panels, the norm-like scalars and the eigenvalues (default: log norms, linear eigenvalues)")
log_derivative_option = click.option("--log-derivative/--no-log-derivative", default=False, help="also plot the step-to-step ratio of the max tangent / df/ds_t norm, with |1 - eig|")
show_option = click.option("--show/--no-show", default=True, help="show the plots")


def show_plots(show: bool):
    if show:
        import matplotlib.pyplot as plt
        plt.show()


@click.group(help=__doc__)
def main():
    pass


@main.command(name="all", help="Redraw the plots of DIR/full_model, DIR/magic* and DIR/rayleigh of each DIR from their saved metrics.")
@click.argument("exp_dirs", type=click.Path(path_type=Path, file_okay=False), nargs=-1)
@yscale_option
@log_derivative_option
@show_option
def replot_all(exp_dirs, yscale, log_derivative, show):
    for exp_dir in exp_dirs:
        replot_experiment(exp_dir, yscale, log_derivative)
    show_plots(show)


@main.command(help="Plot the steps from --start to --start + --n-steps (of the config) of DIR/full_model and DIR/magic* into "
                   "<dir>/metrics_zoom_<start>_<end>[_norms...].png, next to the full plots.")
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--start", type=int, required=True, help="first step plotted")
@click.option("--n-steps", type=int, required=True, help="steps plotted from --start")
@yscale_option
@log_derivative_option
@show_option
def zoom(exp_dir, start, n_steps, yscale, log_derivative, show):
    replot_experiment(exp_dir, yscale, log_derivative, rayleigh=False, steps=(start, start + n_steps))
    show_plots(show)


if __name__ == "__main__":
    main()
