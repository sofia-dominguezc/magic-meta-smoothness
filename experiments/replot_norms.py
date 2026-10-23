"""
Redraw <dir>/metrics.png (scalars) and <dir>/metrics_norms.png from the saved <dir>/metrics.pt of DIR/full_model,
DIR/logged, DIR/magic and DIR/rayleigh (e.g. after a change of Logger._plot_scalars / _plot_norms), without retraining:
  python -m experiments.replot_norms DIR [DIR ...]
--yscale: of the full model's and logged norms (the MAGIC norms df/ds_t are always log).
--log-derivative: also <dir>/metrics_norms_dlog.png, for the "tangent/..." (logged) and "delta" (magic, rayleigh) panels: the
growth per step of y_n = max op norm over the panel's tensors (y_{n+1} / y_n for the tangent, y_n / y_{n+1} for df/ds_t;
linear scale) with |1 - eig| for the eigenvalues logged by train_logged.py.
"""

from collections import defaultdict
from pathlib import Path

import click
import torch

from main import Experiment
from src.logging import Logger


def replot(metrics_path: Path, name: str, yscale=None, eos: float | None = None, log_derivative: bool = False, eigs: dict | None = None,
           smooth: bool = True):
    """`eos`: the stability threshold drawn in the "eig" panel. `eigs`: metrics {"eig/i": {"step", "value"}} drawn as |1 - eig|
    in the derivative plot (default: those of `metrics_path`)."""
    metrics = torch.load(metrics_path, weights_only=True)
    eigs = {k: v for k, v in (metrics if eigs is None else eigs).items() if k.startswith("eig/")}
    panels: dict[str, list[str]] = defaultdict(list)
    for k in metrics:
        panels[k.rsplit("/", 1)[0] if "/" in k else k].append(k)
    norms = {p: names for p, names in panels.items() if "." in names[0].rsplit("/", 1)[-1]}  # lines are tensor names
    scalars = {p: names for p, names in panels.items() if p not in norms}
    logger = Logger(name, smooth=smooth)
    logger.eos = eos
    if scalars:
        logger._plot_scalars(metrics, scalars, metrics_path.with_suffix(".png"))
    if norms:
        logger._plot_norms(metrics, norms, metrics_path.with_name(f"{metrics_path.stem}_norms.png"), yscale=yscale)
    derivs = {p: names for p, names in norms.items() if p.startswith(("tangent", "delta"))}  # directions propagated by the training
    if derivs and log_derivative:
        fig, axes = logger._grid(len(derivs), ncols=2, size=(10, 6))
        for ax, (panel, names) in zip(axes, derivs.items()):
            steps = metrics[names[0]]["step"]
            assert all(torch.equal(metrics[k]["step"], steps) for k in names)
            order = steps.argsort()  # MAGIC logs the steps in reverse
            steps, y = steps[order], torch.stack([metrics[k]["value"][order] for k in names]).max(0).values  # max op norm over the tensors
            backward = panel.startswith("delta")  # df/ds_t is propagated backward: its growth per step is y_n / y_(n+1)
            ratio, label = (y[:-1] / y[1:], "y_n / y_(n+1)") if backward else (y[1:] / y[:-1], "y_(n+1) / y_n")
            ax.plot(steps[:-1].numpy(), ratio.numpy(), lw=1, color="C0", label=f"{label}, y = max op norm")
            for i, k in enumerate(sorted(eigs, key=lambda k: int(k.split("/")[1]), reverse=True)):
                ax.plot(eigs[k]["step"].numpy(), (1 - eigs[k]["value"]).abs().numpy(), "o-", ms=3, lw=1, color=f"C{i + 1}", label=f"|1 - {k}|")
            ax.axhline(1, color="gray", lw=0.5)
            ax.set(title=panel, xlabel="step")
            ax.legend(fontsize=7)
        fig.savefig(metrics_path.with_name(f"{metrics_path.stem}_norms_dlog.png"), dpi=150)
    click.echo(f"{metrics_path.parent}: scalars {list(scalars)}, norms {list(norms)}")


@click.command(help=__doc__)
@click.argument("exp_dirs", type=click.Path(path_type=Path, file_okay=False), nargs=-1)
@click.option("--yscale", type=str, default=None, help="of the full model's and logged norms (the MAGIC norms are always log)")
@click.option("--log-derivative/--no-log-derivative", default=False, help="also plot the step-to-step ratio of the max tangent / df/ds_t norm, with |1 - eig|")
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dirs, yscale, log_derivative, show):
    for exp_dir in exp_dirs:
        exp = Experiment(exp_dir, "cpu")
        eos = exp.init_state.stability_threshold
        eigs = torch.load(exp_dir / "logged" / "metrics.pt", weights_only=True) if (exp_dir / "logged" / "metrics.pt").exists() else {}
        for sub, name, kwargs in [
            ("full_model", exp.name, dict(yscale=yscale, eos=eos)),
            ("logged", exp.name, dict(yscale=yscale, eos=eos, log_derivative=log_derivative)),
            ("magic", f"MAGIC: df/ds_t and df/dw_t; {exp.name}", dict(yscale="log", log_derivative=log_derivative, eigs=eigs, smooth=False)),
            ("rayleigh", f"rayleigh: final Hessian eigenvectors along training, MAGIC's df/ds_t; {exp.name}",
             dict(yscale="log", log_derivative=log_derivative, eigs=eigs, smooth=False)),
        ]:
            if (exp_dir / sub / "metrics.pt").exists():
                replot(exp_dir / sub / "metrics.pt", name, **kwargs)

    if show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
