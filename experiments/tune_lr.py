"""
Train with several learning rates, pick the lowest final test loss and save it to DIR/config.json if it has no lr yet:
  python -m experiments.tune_lr DIR [--lrs 0.01,0.02,...] [--steps n | n1,n2,... (one per lr)] [--magic] [--device cpu] [--no-show]

Saves DIR/lr_tuning.json, the logged curves of every run (DIR/lr_tuning_runs.pt: {"train": {lr: Logger.metrics()}, "magic": ...}) and the plots, one line per lr in each panel:
- DIR/lr_tuning_curves.png: the training logs (loss, metrics, test loss / metrics, total norms of the params and grads);
  DIR/lr_tuning_curves_flow.png (--flow-time): the same against the gradient flow time lr * step
- DIR/lr_tuning_magic.png (--magic): MAGIC on one test subset per lr: the total L2 norms of df/ds_t per state group and |df/dw_t|
- DIR/lr_tuning.png: the min test loss and max test acc over the training and the step of the latter (of the model and, with --ema-gamma / --ema-decay, its
  EMAs) vs lr and, with --magic, the explosion rate max_t |df/ds_t| / |df/ds_T| of the backward (over its finite steps: a run
  whose |df/ds_t| overflowed is marked, and its step saved as "overflow_step")
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

from main import Experiment
from src.magic import TrainLog, backprop_logged, to_device
from src.logging import Logger, ema_smooth, fig_title

Metrics = dict[str, dict[str, Tensor]]  # Logger.metrics()


def plot_runs(runs: dict[float, Metrics], names: list[str], title: str, path: Path, log_names: Collection[str] = (), smooth: bool = True,
              flow_time: bool = False):
    """One panel per metric name, one line per lr (viridis, light to dark = small to large lr); `log_names`: log-scale panels;
    `flow_time`: against the gradient flow time lr * step instead of the step"""
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
            time = steps * lr if flow_time else steps
            if smooth and per_step and len(steps) >= 20 and name not in log_names:
                ax.plot(time, arr, "-", alpha=0.25, lw=0.8, color=colors[lr])
                ax.plot(time, ema_smooth(steps, arr), "-", lw=1.2, color=colors[lr], label=f"{lr:g}")
            else:
                ax.plot(time, arr, "-" if per_step else "o-", ms=3, lw=1, color=colors[lr], label=f"{lr:g}")
        ax.set(title=name, xlabel="lr * step" if flow_time else "step", yscale="log" if name in log_names else "linear")
        ax.title.set_fontsize(9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="lr", loc="outside right center", fontsize=7, title_fontsize=8, frameon=False)
    fig_title(fig, title)
    fig.savefig(path, dpi=150)


def d_state_totals(metrics: Metrics) -> tuple[Tensor, Tensor]:
    """|df/ds_t| over all the state groups at each logged step t, in order: (steps [n], sqrt(sum of the squared
    "d_state/<group>/0" norms) [n])"""
    groups = [m for name, m in metrics.items() if name.startswith("d_state/") and name.endswith("/0")]
    order = groups[0]["step"].argsort()
    return groups[0]["step"][order], torch.stack([m["value"][order] for m in groups]).pow(2).sum(0).sqrt()


def explosion(metrics: Metrics) -> tuple[float, int | None]:
    """(the explosion rate max_t |df/ds_t| / |df/ds_{T-1}| of a MAGIC backward over its finite steps, the last step t at which
    |df/ds_t| overflowed (None if it never did))"""
    steps, totals = d_state_totals(metrics)
    finite = totals.isfinite()
    overflow = None if finite.all() else int(steps[~finite].max())
    return (totals[finite].max() / totals[-1]).item(), overflow


def tune_lr(
    exp_dir: Path, lrs: list[float], steps: int | list[int] | None = None, device: str = "cpu", compile: bool = False, show: bool = True,
    magic: bool = False, n_test: int = 1, ema_gamma: float | None = None, ema_decay: float | None = None,
    eval_every: int | None = None, full_test: bool = False, flow_time: bool = False,
) -> float:
    """`steps`: of every run, or one per lr; `full_test`: every test eval on the whole test set (default: 1000 random examples,
    the whole set at the last step). The test acc of each run is the max over its evals, of the final model and the EMAs."""
    exp = Experiment(exp_dir, device, compile)
    config_lr = exp.cfg.lr
    exp = replace(exp, cfg=replace(exp.cfg, lr=lrs[0]))  # the config may have no lr yet
    steps = exp.cfg.n_steps if steps is None else steps
    steps_per_lr = [n * exp.substeps for n in (steps if isinstance(steps, list) else [steps] * len(lrs))]
    assert len(steps_per_lr) == len(lrs), f"{len(steps_per_lr)} step counts for {len(lrs)} lrs"
    all_test = torch.arange(len(exp.test_ds))
    acc_models = {"": "model", **({"/ema": f"EMA (gamma {ema_gamma:g})"} if ema_gamma is not None else {}),
                  **({"/ema_short": f"EMA (decay {ema_decay:g})"} if ema_decay is not None else {})}  # "test_acc<suffix>" -> label
    losses, explosions, overflows = [], [], []
    max_accs: dict[str, list[float]] = {suffix: [] for suffix in acc_models}
    max_acc_steps: dict[str, list[int]] = {suffix: [] for suffix in acc_models}
    min_losses: dict[str, list[float]] = {suffix: [] for suffix in acc_models}
    train_runs: dict[float, Metrics] = {}
    magic_runs: dict[float, Metrics] = {}
    for lr, steps in zip(tqdm(lrs, desc="learning rates"), steps_per_lr):
        init_state = replace(exp.init_state, lr=replace(exp.init_state.lr, lr=lr))
        logger = Logger(f"lr {lr:g}", smooth=False, total_norms=True, step_scale=exp.substeps)
        train_log = TrainLog(magic=exp.magic, logger=logger, test_loader=exp.test_loader, eval_every=eval_every or max(1, steps // 20),
                             n_test=None if full_test else min(len(all_test), 1000), ema_gamma=ema_gamma, ema_decay=ema_decay)
        final = train_log.train(init_state, n_steps=steps)
        train_runs[lr] = logger.metrics()
        loss, metrics = exp.objective.eval_loss({k: p.to(device) for k, p in final.params.items()}, exp.test_loader.iter_batches(all_test))
        losses.append(loss.item() if math.isfinite(loss.item()) else math.inf)
        msg = f"  lr {lr:g}: test loss {losses[-1]:.4f}, final acc {metrics['acc'].item():.3f}, max acc"
        for suffix, label in acc_models.items():
            accs = train_runs[lr][f"test_acc{suffix}"]
            min_losses[suffix].append(train_runs[lr][f"test_loss{suffix}"]["value"].min().item())
            max_accs[suffix].append(accs["value"].max().item())
            max_acc_steps[suffix].append(int(accs["step"][accs["value"].argmax()]))
            msg += f" {label} {max_accs[suffix][-1]:.3f} (step {max_acc_steps[suffix][-1]})"
        if magic:
            logger = Logger(f"MAGIC lr {lr:g}", smooth=False, total_norms=True, step_scale=exp.substeps)
            eval_set = exp.test_ds.examples(exp.test_ds.subset(n_test, index=0))
            d_state = final.cotangent(exp.objective.eval_grads(to_device(final.params, device), [eval_set])[1])
            backprop_logged(exp.magic, init_state, d_state, [eval_set], logger, n_steps=steps)
            magic_runs[lr] = logger.metrics()
            rate, overflow = explosion(magic_runs[lr])
            explosions.append(rate)
            overflows.append(overflow)
            msg += f", explosion rate {rate:.3g}" + ("" if overflow is None else f" (|df/ds_t| overflowed at step {overflow})")
        click.echo(msg)
    best = lrs[min(range(len(lrs)), key=lambda i: losses[i])]
    if config_lr is None:
        click.echo(f"best lr: {best:g} (saved to {exp_dir / 'config.json'})")
        (exp_dir / "config.json").write_text(json.dumps(asdict(replace(exp.cfg, lr=best)), indent=2))
    else:
        click.echo(f"best lr: {best:g} (config keeps lr {config_lr:g})")
    summary = {"steps": steps_per_lr, "lrs": lrs, "test_loss": losses, "best": best, "ema_gamma": ema_gamma, "ema_decay": ema_decay,
               **{f"max_test_acc{suffix.replace('/', '_')}": values for suffix, values in max_accs.items()},
               **{f"max_test_acc_step{suffix.replace('/', '_')}": values for suffix, values in max_acc_steps.items()},
               **{f"min_test_loss{suffix.replace('/', '_')}": values for suffix, values in min_losses.items()}}
    if magic:
        summary["explosion_rate"], summary["overflow_step"] = explosions, overflows
    (exp_dir / "lr_tuning.json").write_text(json.dumps(summary, indent=2))
    torch.save({"train": train_runs, "magic": magic_runs}, exp_dir / "lr_tuning_runs.pt")

    steps_label = f"{steps_per_lr[0]} steps" if len(set(steps_per_lr)) == 1 else f"{min(steps_per_lr)}-{max(steps_per_lr)} steps"
    title = f"{exp.cfg.model} on {exp.cfg.dataset}, {exp.cfg.optimizer}, {steps_label}"
    norm_names = {"params", "grads"}
    names = [n for n in dict.fromkeys(n for m in train_runs.values() for n in m)
             if n not in ("lr", "gpu_gb", "gpu_util") and ("/ema" not in n or n.startswith(("test_loss/", "test_acc/")))]
    names.sort(key=lambda n: (n in norm_names, n.startswith("test_")))  # train metrics, test metrics, norms (stable)
    plot_runs(train_runs, names, title, exp_dir / "lr_tuning_curves.png", log_names=norm_names)
    if flow_time:
        plot_runs(train_runs, names, title, exp_dir / "lr_tuning_curves_flow.png", log_names=norm_names, flow_time=True)
    if magic:
        names = list(dict.fromkeys(n for m in magic_runs.values() for n in m))
        plot_runs(magic_runs, names, f"MAGIC on {n_test} test examples; {title}",
                  exp_dir / "lr_tuning_magic.png",
                  log_names=set(names), smooth=False)

    plot_summary(lrs, min_losses, max_accs, max_acc_steps, acc_models, explosions if magic else None, overflows, best, title, exp_dir / "lr_tuning.png")
    if show:
        plt.show()
    plt.close("all")
    return best


def plot_summary(lrs: list[float], min_losses: dict[str, list[float]], max_accs: dict[str, list[float]], max_acc_steps: dict[str, list[int]],
                 acc_models: dict[str, str], explosions: list[float] | None, overflows: list[int | None], best: float, title: str, path: Path) -> plt.Figure:
    """lr_tuning.png: min test loss, max test acc and its step (one line per model of `acc_models`: "test_<metric><suffix>" -> label)
    and, if given, the explosion rate (the points where the backward overflowed marked) vs lr"""
    finals = [(min_losses[""], "min test loss", "linear"), (max_accs[""], "max test acc", "linear"), (max_acc_steps[""], "step of the max test acc", "log")]
    if explosions is not None:
        finals.append((explosions, "explosion rate max_t |df/ds_t| / |df/ds_T|", "log"))
    fig, axes = plt.subplots(1, len(finals), figsize=(4.5 * len(finals), 3.6), layout="constrained")
    for ax, (values, name, yscale) in zip(axes, finals):
        ax.plot(lrs, values, "o-", label=acc_models[""])
        per_model = {"min test loss": min_losses, "max test acc": max_accs, "step of the max test acc": max_acc_steps}.get(name)
        if per_model is not None and len(acc_models) > 1:
            for (suffix, label), marker in zip(list(acc_models.items())[1:], ("s--", "^:")):
                ax.plot(lrs, per_model[suffix], marker, label=label)
            ax.legend(fontsize=7, frameon=False)
        if name.startswith("explosion") and any(t is not None for t in overflows):
            overflowed = [(lr, rate) for lr, rate, t in zip(lrs, values, overflows) if t is not None]
            ax.plot(*zip(*overflowed), "rx", ms=8, label="overflowed (rate before)")
            ax.legend(fontsize=7, frameon=False)
        ax.axvline(best, color="gray", ls="--", lw=0.8)
        ax.set(xscale="log", yscale=yscale, xlabel="lr", ylabel=name)
        ax.set_xticks(lrs, [f"{lr:g}" for lr in lrs])
        ax.tick_params(which="minor", labelbottom=False)
    fig_title(fig, title)
    fig.savefig(path, dpi=150)
    return fig


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--lrs", type=str, default="0.01,0.02,0.05,0.1,0.2,0.5,1", show_default=True, help="comma-separated learning rates")
@click.option("--steps", type=str, default=None, help="train this many steps, or a comma-separated count per lr (default: n_steps)")
@click.option("--magic/--no-magic", default=False, help="also run MAGIC on one test subset per lr (norms of df/ds_t, explosion rate)")
@click.option("--n-test", type=int, default=1, show_default=True, help="test examples of the MAGIC eval subset")
@click.option("--ema-gamma", type=float, default=None, help="also report the final test acc of the power-function EMA of the params with this exponent (see main.py train --ema-gamma)")
@click.option("--ema-decay", type=float, default=None, help="also report the max test acc of the short-term EMA of the params with this decay (see main.py train --ema-decay)")
@click.option("--eval-every", type=int, default=None, help="log the test loss every this many steps (default: steps / 20)")
@click.option("--full-test/--no-full-test", default=False, help="every test eval on the whole test set (default: 1000 random examples, the whole set at the last step)")
@click.option("--flow-time/--no-flow-time", default=False, help="also plot the curves against the gradient flow time lr * step (lr_tuning_curves_flow.png)")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--compile/--no-compile", default=False, help="torch.compile the training passes")
@click.option("--show/--no-show", default=True, help="show the plot")
def main(exp_dir, lrs, steps, magic, n_test, ema_gamma, ema_decay, eval_every, full_test, flow_time, device, compile, show):
    if steps is not None:
        steps = [int(n) for n in steps.split(",")]
        steps = steps[0] if len(steps) == 1 else steps
    tune_lr(exp_dir, [float(v) for v in lrs.split(",")], steps, device, compile, show, magic=magic, n_test=n_test,
            ema_gamma=ema_gamma, ema_decay=ema_decay, eval_every=eval_every, full_test=full_test, flow_time=flow_time)


if __name__ == "__main__":
    main()
