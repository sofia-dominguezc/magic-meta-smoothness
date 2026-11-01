"""
Train with several learning rates, pick the lowest final test loss and save it to DIR/config.json:
  python -m experiments.tune_lr DIR [--lrs 0.01,0.02,...] [--steps n] [--magic] [--device cpu] [--no-show]

Saves DIR/lr_tuning.json and the plots, one line per lr in each panel:
- DIR/lr_tuning_curves.png: the training logs (loss, metrics, test loss / metrics, total norms of the params and grads)
- DIR/lr_tuning_magic.png (--magic): MAGIC on one test subset per lr: the total L2 norms of df/ds_t per state group and |df/dw_t|
- DIR/lr_tuning.png: final test loss and acc vs lr and, with --magic, the explosion rate |df/ds_0| / |df/ds_T| of the backward
"""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from typing import Collection

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from main import HESSIAN, Experiment, parse_variant, train_logged
from src.logging import Logger, ema_smooth, fig_title

Metrics = dict[str, dict[str, Tensor]]  # Logger.metrics()


def plot_runs(runs: dict[float, Metrics], names: list[str], title: str, path: Path, log_names: Collection[str] = (), smooth: bool = True):
    """One panel per metric name, one line per lr (viridis, light to dark = small to large lr); `log_names`: log-scale panels"""
    if not names:
        return
    lrs = list(runs)
    colors = dict(zip(lrs, plt.cm.viridis(np.linspace(0.9, 0, len(lrs)))))
    ncols = min(4, len(names))
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.4 * nrows), squeeze=False, layout="constrained")
    for ax in axes.flat[len(names):]:
        fig.delaxes(ax)
    for ax, name in zip(axes.flat, names):
        for lr, metrics in runs.items():
            if name not in metrics:
                continue
            steps, arr = metrics[name]["step"].numpy(), metrics[name]["value"].numpy()
            order = np.argsort(steps, kind="stable")  # (MAGIC logs the steps in reverse)
            steps, arr = steps[order], arr[order]
            per_step = len(steps) > 1 and steps[1] - steps[0] == 1
            if smooth and per_step and len(steps) >= 20 and name not in log_names:
                ax.plot(steps, arr, "-", alpha=0.25, lw=0.8, color=colors[lr])
                ax.plot(steps, ema_smooth(steps, arr), "-", lw=1.2, color=colors[lr], label=f"{lr:g}")
            else:
                ax.plot(steps, arr, "-" if per_step else "o-", ms=3, lw=1, color=colors[lr], label=f"{lr:g}")
        ax.set(title=name, xlabel="step", yscale="log" if name in log_names else "linear")
        ax.title.set_fontsize(9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="lr", loc="outside right center", fontsize=7, title_fontsize=8, frameon=False)
    fig_title(fig, title)
    fig.savefig(path, dpi=150)


def delta_total(metrics: Metrics, step: int) -> float:
    """|df/ds_step| over all the state groups: sqrt(sum of the squared "delta/<group>/0" norms logged at `step`)"""
    total = 0.0
    for name, m in metrics.items():
        if name.startswith("delta/") and name.endswith("/0"):
            at = m["value"][m["step"] == step]
            total += at.pow(2).sum().item()
    return total**0.5


def tune_lr(
    exp_dir: Path, lrs: list[float], steps: int | None = None, device: str = "cpu", compile: bool = False, show: bool = True,
    magic: bool = False, n_test: int = 1, variant: str = HESSIAN,
) -> float:
    exp = Experiment(exp_dir, device, compile)
    exp = replace(exp, cfg=replace(exp.cfg, lr=lrs[0]))  # the config may have no lr yet
    steps = (exp.cfg.n_steps if steps is None else steps) * exp.substeps
    all_test = torch.arange(len(exp.test_ds))
    losses, accs, explosions = [], [], []
    train_runs: dict[float, Metrics] = {}
    magic_runs: dict[float, Metrics] = {}
    for lr in tqdm(lrs, desc="learning rates"):
        init_state = replace(exp.init_state, lr=replace(exp.init_state.lr, lr=lr))
        logger = Logger(f"lr {lr:g}", smooth=False, total_norms=True, step_scale=exp.substeps)
        final = train_logged(exp.magic, init_state, n_steps=steps, test_loader=exp.test_loader, logger=logger,
                             eval_every=max(1, steps // 20), n_test=min(len(all_test), 1000), ema_decay=None)
        train_runs[lr] = logger.metrics()
        loss, metrics = exp.magic.eval_loss({k: p.to(device) for k, p in final.params.items()}, exp.test_loader, all_test)
        losses.append(loss.item() if math.isfinite(loss.item()) else math.inf)
        accs.append(metrics["acc"].item())
        msg = f"  lr {lr:g}: test loss {losses[-1]:.4f}, acc {accs[-1]:.3f}"
        if magic:
            logger = Logger(f"MAGIC lr {lr:g}", smooth=False, total_norms=True, step_scale=exp.substeps)
            eval_set = exp.test_ds.examples(exp.test_ds.subset(n_test, index=0))
            gn_below, skip_below = parse_variant(variant, replace(exp.cfg, n_steps=steps))
            exp.magic.backprop(init_state, n_steps=steps, eval_sets=[eval_set], gn_below=gn_below, skip_below=skip_below, log=logger)
            magic_runs[lr] = logger.metrics()
            explosions.append(delta_total(magic_runs[lr], 0) / delta_total(magic_runs[lr], steps))
            msg += f", explosion rate {explosions[-1]:.3g}"
        click.echo(msg)
    best = lrs[min(range(len(lrs)), key=lambda i: losses[i])]
    click.echo(f"best lr: {best:g} (saved to {exp_dir / 'config.json'})")
    (exp_dir / "config.json").write_text(json.dumps(asdict(replace(exp.cfg, lr=best)), indent=2))
    summary = {"steps": steps, "lrs": lrs, "test_loss": losses, "test_acc": accs, "best": best}
    if magic:
        summary["explosion_rate"] = explosions
    (exp_dir / "lr_tuning.json").write_text(json.dumps(summary, indent=2))

    title = f"{exp.cfg.model} on {exp.cfg.dataset}, {exp.cfg.optimizer}, {steps} steps"
    norm_names = {"params", "grads"}
    names = [n for n in dict.fromkeys(n for m in train_runs.values() for n in m) if n not in ("lr", "gpu_gb")]
    names.sort(key=lambda n: (n in norm_names, n.startswith("test_")))  # train metrics, test metrics, norms (stable)
    plot_runs(train_runs, names, title, exp_dir / "lr_tuning_curves.png", log_names=norm_names)
    if magic:
        names = list(dict.fromkeys(n for m in magic_runs.values() for n in m))
        plot_runs(magic_runs, names, f"MAGIC{'' if variant == HESSIAN else f' ({variant})'} on {n_test} test examples; {title}",
                  exp_dir / "lr_tuning_magic.png",
                  log_names=set(names), smooth=False)

    finals = [(losses, "test loss", "linear"), (accs, "test acc", "linear")]
    if magic:
        finals.append((explosions, "explosion rate |df/ds_0| / |df/ds_T|", "log"))
    fig, axes = plt.subplots(1, len(finals), figsize=(4.5 * len(finals), 3.6), layout="constrained")
    for ax, (values, name, yscale) in zip(axes, finals):
        ax.plot(lrs, values, "o-")
        ax.axvline(best, color="gray", ls="--", lw=0.8)
        ax.set(xscale="log", yscale=yscale, xlabel="lr", ylabel=name)
        ax.set_xticks(lrs, [f"{lr:g}" for lr in lrs])
        ax.tick_params(which="minor", labelbottom=False)
    fig_title(fig, title)
    fig.savefig(exp_dir / "lr_tuning.png", dpi=150)
    if show:
        plt.show()
    plt.close("all")
    return best


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--lrs", type=str, default="0.01,0.02,0.05,0.1,0.2,0.5,1,2,5", show_default=True, help="comma-separated learning rates")
@click.option("--steps", type=int, default=None, help="train this many steps (default: n_steps)")
@click.option("--magic/--no-magic", default=False, help="also run MAGIC on one test subset per lr (norms of df/ds_t, explosion rate)")
@click.option("--n-test", type=int, default=1, show_default=True, help="test examples of the MAGIC eval subset")
@click.option("--variant", type=str, default=HESSIAN, show_default=True, help="the MAGIC variant (see main.py magic --variant)")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--compile/--no-compile", default=False, help="torch.compile the training passes")
@click.option("--show/--no-show", default=True, help="show the plot")
def main(exp_dir, lrs, steps, magic, n_test, variant, device, compile, show):
    tune_lr(exp_dir, [float(v) for v in lrs.split(",")], steps, device, compile, show, magic=magic, n_test=n_test, variant=variant)


if __name__ == "__main__":
    main()
