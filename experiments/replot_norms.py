"""
Redraw <dir>/metrics.png (scalars) and <dir>/metrics_norms.png from the saved <dir>/metrics.pt of DIR/full_model and
DIR/magic (e.g. after a change of Logger._plot_scalars / _plot_norms), without retraining:
  python -m experiments.replot_norms DIR [DIR ...]
"""

from collections import defaultdict
from pathlib import Path

import click
import torch

from main import Experiment
from src.logging import Logger


def replot(metrics_path: Path, name: str, yscale=None, eos: float | None = None):
    """`eos`: the stability threshold drawn in the "eig" panel."""
    metrics = torch.load(metrics_path, weights_only=True)
    panels: dict[str, list[str]] = defaultdict(list)
    for k in metrics:
        panels[k.rsplit("/", 1)[0] if "/" in k else k].append(k)
    norms = {p: names for p, names in panels.items() if "." in names[0].rsplit("/", 1)[-1]}  # lines are tensor names
    scalars = {p: names for p, names in panels.items() if p not in norms}
    logger = Logger(name)
    logger.eos = eos
    if scalars:
        logger._plot_scalars(metrics, scalars, metrics_path.with_suffix(".png"))
    if norms:
        logger._plot_norms(metrics, norms, metrics_path.with_name(f"{metrics_path.stem}_norms.png"), yscale=yscale)
    click.echo(f"{metrics_path.parent}: scalars {list(scalars)}, norms {list(norms)}")


@click.command(help=__doc__)
@click.argument("exp_dirs", type=click.Path(path_type=Path, file_okay=False), nargs=-1)
@click.option("--yscale", type=str, default=None)
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dirs, yscale, show):
    for exp_dir in exp_dirs:
        exp = Experiment(exp_dir, "cpu")
        replot(exp_dir / "full_model" / "metrics.pt", exp.name, yscale=yscale, eos=exp.init_state.stability_threshold)
        ns = {len(torch.load(p, weights_only=True)["indices"]) for p in (exp_dir / "magic").glob("*/linearization.pt")}
        if (exp_dir / "magic" / "metrics.pt").exists():
            replot(exp_dir / "magic" / "metrics.pt", f"MAGIC: norms of df/ds_t, f = loss on {', '.join(map(str, sorted(ns)))} test examples; {exp.name}", yscale=yscale)

    if show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
