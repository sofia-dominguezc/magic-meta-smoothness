"""
Train the full model at several learning rates, one experiment dir DIR/lr<lr> each (DIR/config.json is their template), and plot
them together. Rerunning with more learning rates trains only the new ones: an lr whose DIR/lr<lr>/full_model/metrics.pt exists
(a trained dir, or a symlink to one made with --run) is reused; delete the dir to retrain it. The lrs train from the largest down.
  python -m experiments.lr_sweep DIR --lrs 0.01,0.02,0.05 [--steps n | n1,n2,...] [--seeds n] [--run EXP_DIR]... [--point EXP_DIR]... [--checkpoint-every k] [--device cuda] [--no-show]
`--seeds n`: each lr trains the models of init seeds init_seed, ..., init_seed + n - 1 (of the template) vmapped together, in
one run (`n_seeds` of its config; DIR/lr<lr>/full_model/metrics.png: the mean of the seeds, +- 2 std dashed; metrics_seeds.png
next to it, for every run of several seeds: one line per seed, see `plot_seed_curves`); a reused run of
fewer seeds is an error, of more: its first n seeds. Every statistic is of the EMA of the params of decay 0.9 ("test_<name>/ema_short",
the power-function EMA is not trained), per seed; the best seed of an lr is the one of the highest max test acc. Saves:
- DIR/lr_tuning.json: per lr, per seed: the min test loss, the max test acc and its step, their mean over the seeds and the
  interval of the middle 95% of the seeds (2.5th to 97.5th percentile); the best seed, the best lr (of the highest mean max test acc)
- DIR/lr_tuning_runs.pt: the metrics of every run, all seeds
- DIR/lr_tuning.png: min test loss, max test acc and its step vs lr: each seed light blue, their mean with the interval of their middle 95%
- DIR/lr_tuning_curves.png: the curves of the best seed of each lr against the gradient flow time lr * step, log scale. Only the
  metrics every run logged are plotted; the per-tensor norms `main.py train` logs ("params/<tensor>", "grads/<tensor>") are
  collapsed to one "params" / "grads" value, the root sum of their squares
`--point`: other trained experiment dirs (e.g. the central flow at some lr) drawn as one point per panel of lr_tuning.png at their
lr, labelled by their optimizer (of their model, not an EMA; their steps are config steps: the central flow's logged substeps
divided by its substeps per step, read off a finished run as its last logged step / n_steps, else the current SUBSTEPS).
"""

from dataclasses import asdict, replace
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.tune_lr import Metrics, plot_runs
from main import Experiment, ExperimentConfig, train
from src.logging import ema_smooth, fig_title

EMA = "/ema_short"  # the metrics of the EMA of the params of decay EMA_DECAY
EMA_DECAY = 0.9  # the default of `main.py train`
SEED_PANELS = ["loss", "acc", "margin", "grad_displacement", "params", "grads", f"test_loss{EMA}", f"test_acc{EMA}", f"test_margin{EMA}"]
TRAIN_HALF_LIFE = 100  # steps, of the smoothing of the train metrics of metrics_seeds.png


def load_config(exp_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(**json.loads((exp_dir / "config.json").read_text()))


def total_norms(metrics: Metrics) -> Metrics:
    """The metrics with the per-tensor norms "params/<tensor>" and "grads/<tensor>" replaced by one "params" / "grads", the root
    sum of their squares (the step of the first tensor)"""
    totals: Metrics = {}
    for quantity in ("params", "grads"):
        tensors = [m for name, m in metrics.items() if name.startswith(f"{quantity}/")]
        if tensors:
            totals[quantity] = {"step": tensors[0]["step"], "value": torch.stack([m["value"] for m in tensors]).pow(2).sum(0).sqrt()}
    return {**{name: m for name, m in metrics.items() if not name.startswith(("params/", "grads/"))}, **totals}


def first_seeds(metrics: Metrics, n_seeds: int) -> Metrics:
    """Each value [steps, n_seeds]: of the first `n_seeds` seeds of a run (a single-seed run's [steps] as one seed)"""
    return {name: {"step": m["step"], "value": (m["value"] if m["value"].ndim == 2 else m["value"][:, None])[:, :n_seeds]} for name, m in metrics.items()}


def seed_stats(metrics: Metrics) -> dict[str, list]:
    """Per seed: the min test loss, the max test acc and its step (of the EMA)"""
    loss, acc = metrics[f"test_loss{EMA}"], metrics[f"test_acc{EMA}"]
    return {"min_test_loss": loss["value"].min(0).values.tolist(), "max_test_acc": acc["value"].max(0).values.tolist(),
            "max_test_acc_step": acc["step"][acc["value"].argmax(0)].tolist()}


def plot_seed_curves(metrics: Metrics, seeds: list[int], title: str, path: Path):
    """metrics_seeds.png: one line per seed of the `SEED_PANELS` logged: the train metrics EMA-smoothed (half-life TRAIN_HALF_LIFE),
    the test metrics of the EMA model as logged"""
    names = [name for name in SEED_PANELS if name in metrics]
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(seeds)))
    fig, axes = plt.subplots(-(-len(names) // 3), 3, figsize=(13.5, 3.4 * -(-len(names) // 3)), squeeze=False, layout="constrained")
    for ax in axes.flat[len(names):]:
        fig.delaxes(ax)
    for ax, name in zip(axes.flat, names):
        steps, values = metrics[name]["step"].numpy(), metrics[name]["value"].numpy()
        test = name.startswith("test_")
        for k, seed in enumerate(seeds):
            ax.plot(steps, values[:, k] if test else ema_smooth(steps, values[:, k], TRAIN_HALF_LIFE), "-", lw=1, color=colors[k], label=str(seed))
        label = name.replace(EMA, f" of the EMA model (decay {EMA_DECAY:g})") if test else f"{name} (EMA-smoothed, half-life {TRAIN_HALF_LIFE})"
        ax.set(title=label, xlabel="step", yscale="log" if name in ("params", "grads") else "linear")
        ax.title.set_fontsize(9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="init seed", loc="outside right center", fontsize=7, title_fontsize=8, frameon=False)
    fig_title(fig, title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mean_interval(values: list[float]) -> tuple[float, float, float]:
    """The mean of the seeds' values and the interval of their middle 95% (2.5th, 97.5th percentiles, interpolated)"""
    low, high = np.percentile(values, [2.5, 97.5])
    return float(np.mean(values)), float(low), float(high)


SUMMARY_PANELS = (("min_test_loss", "min test loss", "linear"), ("max_test_acc", "max test acc", "linear"),
                  ("max_test_acc_step", "step of the max test acc", "log"))


def plot_seeds_summary(lrs: list[float], stats: dict[float, dict[str, list]], best_lr: float, title: str, path: Path) -> plt.Figure:
    """lr_tuning.png: min test loss, max test acc and its step vs lr: each seed light blue, their mean with the interval of their middle 95%"""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), layout="constrained")
    n_seeds = len(stats[lrs[0]]["max_test_acc"])
    for ax, (key, name, yscale) in zip(axes, SUMMARY_PANELS):
        if n_seeds > 1:
            ax.plot(*zip(*[(lr, value) for lr in lrs for value in stats[lr][key]]), "o", ms=3, color="lightblue", label="seeds", zorder=1)
        means, lows, highs = (np.array(values) for values in zip(*[mean_interval(stats[lr][key]) for lr in lrs]))
        label = f"EMA (decay {EMA_DECAY:g})" + (f": mean of {n_seeds} seeds, middle 95% of the seeds" if n_seeds > 1 else "")
        ax.errorbar(lrs, means, yerr=[means - lows, highs - means], fmt="o-", ms=5, capsize=3, color="C0", label=label)
        ax.axvline(best_lr, color="gray", ls="--", lw=0.8)
        ax.set(xscale="log", xlabel="lr", ylabel=name, yscale=yscale)
        ax.set_xticks(lrs, [f"{lr:g}" for lr in lrs], rotation=45)
        ax.tick_params(which="minor", labelbottom=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", fontsize=8, frameon=False)
    fig_title(fig, title)
    fig.savefig(path, dpi=150)
    return fig


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option("--lrs", type=str, required=True, help="comma-separated learning rates")
@click.option("--steps", type=str, default=None, help="train this many steps, or a comma-separated count per lr (default: the template's n_steps)")
@click.option("--seeds", type=int, default=1, show_default=True, help="models per lr, of init seeds init_seed, init_seed + 1, ... of the template, trained vmapped")
@click.option("--run", "runs", type=click.Path(path_type=Path, file_okay=False, exists=True), multiple=True,
              help="a trained experiment dir to reuse: symlinked as DIR/lr<its lr> (repeatable)")
@click.option("--point", "points", type=click.Path(path_type=Path, file_okay=False, exists=True), multiple=True,
              help="a trained experiment dir drawn as a single point per panel of lr_tuning.png (repeatable)")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--compile/--no-compile", default=False, help="torch.compile the training passes")
@click.option("--eval-every", type=int, default=100, show_default=True, help="log the test loss every this many steps")
@click.option("--n-test", type=int, default=0, show_default=True, help="test examples of each test eval, a fresh random subset (0: the whole test set, always at the last step)")
@click.option("--checkpoint-every", type=int, default=0, show_default=True, help="of the newly trained runs (see `main.py train`; for experiments.backprop_sweep)")
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dir: Path, lrs: str, steps: str | None, seeds: int, runs: tuple[Path, ...], points: tuple[Path, ...], device: str, compile: bool,
         eval_every: int, n_test: int, checkpoint_every: int, show: bool):
    template = load_config(exp_dir)
    lr_list = [float(lr) for lr in lrs.split(",")]
    steps_per_lr = [template.n_steps] * len(lr_list) if steps is None else [int(n) for n in steps.split(",")]
    steps_per_lr = steps_per_lr * len(lr_list) if len(steps_per_lr) == 1 else steps_per_lr
    assert len(steps_per_lr) == len(lr_list), f"{len(steps_per_lr)} step counts for {len(lr_list)} lrs"
    lr_list, steps_per_lr = (list(values) for values in zip(*sorted(zip(lr_list, steps_per_lr))))
    for run in runs:
        link = exp_dir / f"lr{load_config(run).lr:g}"
        if not link.exists():
            link.symlink_to(Path("..") / run.resolve().relative_to(exp_dir.resolve().parent) if run.resolve().parent == exp_dir.resolve().parent else run.resolve())
            click.echo(f"linked {link} -> {run}")

    run_metrics: dict[float, Metrics] = {}
    run_steps: dict[float, int] = {}
    run_seeds: dict[float, list[int]] = {}  # the init seed of each model
    unfinished: dict[float, int] = {}  # lr -> its steps so far
    for lr, n_steps in reversed(list(zip(lr_list, steps_per_lr))):  # the largest lr first
        run_dir = exp_dir / f"lr{lr:g}"
        if (run_dir / "full_model" / "metrics.pt").exists():
            cfg = load_config(run_dir)
            if cfg.n_seeds < seeds:
                raise click.UsageError(f"{run_dir} has {cfg.n_seeds} seed(s), not {seeds}: delete it (or the link) to retrain it")
            click.echo(f"lr {lr:g}: reusing {run_dir} ({cfg.n_steps} steps, {cfg.n_seeds} seed(s))")
        else:
            cfg = replace(template, lr=lr, n_steps=n_steps, n_seeds=seeds)
            run_dir.mkdir(exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
            click.echo(f"lr {lr:g}: training {run_dir} for {n_steps} steps, {seeds} seed(s)")
            with click.Context(train) as ctx:
                ctx.invoke(train, exp_dir=run_dir, device=device, compile=compile, eval_every=eval_every, n_test=n_test, ema_gamma=0,
                           ema_decay=EMA_DECAY, checkpoint_every=checkpoint_every, show=False)
        run_seeds[lr] = [cfg.init_seed + k for k in range(seeds)]
        metrics = total_norms(torch.load(run_dir / "full_model" / "metrics.pt", weights_only=True))
        run_steps[lr] = cfg.n_steps
        if not (run_dir / "full_model" / "final.pt").exists():  # still training (or killed): its steps so far
            run_steps[lr] = int(metrics["loss"]["step"].max()) + 1
            unfinished[lr] = run_steps[lr]
            click.echo(f"lr {lr:g}: unfinished, {run_steps[lr]} of {cfg.n_steps} steps so far")
        if cfg.n_seeds > 1:
            plot_seed_curves(metrics, [cfg.init_seed + k for k in range(cfg.n_seeds)],
                             f"{cfg.model} on {cfg.dataset}, {cfg.optimizer} lr {cfg.lr:g}, {cfg.n_seeds} seeds", run_dir / "full_model" / "metrics_seeds.png")
        run_metrics[lr] = first_seeds(metrics, seeds)

    stats = {lr: seed_stats(run_metrics[lr]) for lr in lr_list}
    best_index = {lr: max(range(seeds), key=lambda k: stats[lr]["max_test_acc"][k]) for lr in lr_list}
    best_seed = {lr: run_seeds[lr][best_index[lr]] for lr in lr_list}
    summaries = {lr: {key: mean_interval(values) for key, values in stats[lr].items()} for lr in lr_list}  # (mean, low, high)
    best_lr = max(lr_list, key=lambda lr: summaries[lr]["max_test_acc"][0])
    for lr in lr_list:
        click.echo(f"  lr {lr:g}: {run_steps[lr]} steps, mean of {seeds} seed(s) [middle 95%]: " + ", ".join(
            f"{key.replace('_', ' ')} {mean:.4g} [{low:.4g}, {high:.4g}]" for key, (mean, low, high) in summaries[lr].items()))
    steps_list = [run_steps[lr] for lr in lr_list]
    summary = {"lrs": lr_list, "steps": steps_list, "ema_decay": EMA_DECAY, "seeds": [run_seeds[lr] for lr in lr_list],
               **{key: [stats[lr][key] for lr in lr_list] for key in stats[lr_list[0]]},
               **{f"{key}_mean": [summaries[lr][key][0] for lr in lr_list] for key in stats[lr_list[0]]},
               **{f"{key}_low95": [summaries[lr][key][1] for lr in lr_list] for key in stats[lr_list[0]]},
               **{f"{key}_high95": [summaries[lr][key][2] for lr in lr_list] for key in stats[lr_list[0]]},
               "best_seed": [best_seed[lr] for lr in lr_list], "best": best_lr}
    (exp_dir / "lr_tuning.json").write_text(json.dumps(summary, indent=2))
    torch.save({"train": run_metrics, "magic": {}}, exp_dir / "lr_tuning_runs.pt")

    steps_label = f"{steps_list[0]} steps" if len(set(steps_list)) == 1 else f"{min(steps_list)}-{max(steps_list)} steps"
    title = f"{template.model} on {template.dataset}, {template.optimizer}, {steps_label}" + ("" if seeds == 1 else f", {seeds} seeds") \
        + ("".join(f"\nunfinished: lr {lr:g} ({n} steps so far)" for lr, n in unfinished.items()))
    norm_names = {"params", "grads"}
    best_runs = {lr: {name: {"step": m["step"], "value": m["value"][:, best_index[lr] if m["value"].shape[1] > 1 else 0]}  # (gpu_gb: one value per step)
                      for name, m in run_metrics[lr].items()} for lr in lr_list}
    names = [n for n in best_runs[lr_list[0]] if all(n in m for m in best_runs.values())
             and n not in ("lr", "gpu_gb", "gpu_util") and ("/ema" not in n or n.startswith("test_") and n.endswith(EMA))]
    names.sort(key=lambda n: (n in norm_names, n.startswith("test_")))
    plot_runs(best_runs, names, title + (", best seed per lr" if seeds > 1 else ""), exp_dir / "lr_tuning_curves.png",
              log_names=norm_names, flow_time=True, log_time=True)
    fig = plot_seeds_summary(lr_list, stats, best_lr, title, exp_dir / "lr_tuning.png")
    for point, marker in zip(points, ("r*", "m*", "c*", "y*", "k*", "b*", "yD", "yP")):
        cfg, metrics = load_config(point), torch.load(point / "full_model" / "metrics.pt", weights_only=True)
        accs, done = metrics["test_acc"], (point / "full_model" / "final.pt").exists()
        substeps = round(int(accs["step"][-1]) / cfg.n_steps) if done else Experiment(point).substeps
        values = (metrics["test_loss"]["value"].min().item(), accs["value"].max().item(), int(accs["step"][accs["value"].argmax()]) // substeps)
        label = f"{cfg.optimizer} lr {cfg.lr:g}" + ("" if done else f" ({int(accs['step'][-1]) // substeps} steps so far)")
        click.echo(f"  {label}: min test loss {values[0]:.4f}, max test acc {values[1]:.3f} at step {values[2]}")
        for ax, value in zip(fig.axes, values):
            ax.plot([cfg.lr], [value], marker, ms=12, label=label)
            ax.legend(fontsize=7, frameon=False)  # (the points: per panel)
    if points:
        fig.savefig(exp_dir / "lr_tuning.png", dpi=150)
    click.echo(f"saved {exp_dir / 'lr_tuning_curves.png'} and {exp_dir / 'lr_tuning.png'}")
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
