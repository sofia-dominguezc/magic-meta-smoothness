"""
Train the full model at several learning rates, one experiment dir DIR/lr<lr> each (DIR/config.json is their template), and plot
them together. Rerunning with more learning rates trains only the new ones: an lr whose DIR/lr<lr>/full_model/metrics.pt exists
(a trained dir, or a symlink to one made with --run) is reused; delete the dir to retrain it.
  python -m experiments.lr_sweep DIR --lrs 0.01,0.02,0.05 [--steps n | n1,n2,...] [--run EXP_DIR]... [--point EXP_DIR]... [--device cuda] [--no-show]
Saves DIR/lr_tuning.json, DIR/lr_tuning_runs.pt, DIR/lr_tuning_curves.png and DIR/lr_tuning.png as experiments.tune_lr does; the
test loss of a run is its last logged "test_loss" (the whole test set at the last step of `main.py train`). Only the metrics every
run logged are plotted; the per-tensor norms `main.py train` logs ("params/<tensor>", "grads/<tensor>") are collapsed to one
"params" / "grads" value, the root sum of their squares. `--point`: other trained experiment dirs (e.g. the central flow at some lr)
drawn as one point per panel of lr_tuning.png at their lr, labelled by their optimizer (their EMAs, if logged, are not drawn; their
steps are config steps: the central flow's logged substeps divided by its substeps per step, read off a finished run as its last
logged step / n_steps, else the current SUBSTEPS).
"""

from dataclasses import asdict, replace
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch

from experiments.tune_lr import Metrics, plot_runs, plot_summary
from main import Experiment, ExperimentConfig, train


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


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option("--lrs", type=str, required=True, help="comma-separated learning rates")
@click.option("--steps", type=str, default=None, help="train this many steps, or a comma-separated count per lr (default: the template's n_steps)")
@click.option("--run", "runs", type=click.Path(path_type=Path, file_okay=False, exists=True), multiple=True,
              help="a trained experiment dir to reuse: symlinked as DIR/lr<its lr> (repeatable)")
@click.option("--point", "points", type=click.Path(path_type=Path, file_okay=False, exists=True), multiple=True,
              help="a trained experiment dir drawn as a single point per panel of lr_tuning.png (repeatable)")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--compile/--no-compile", default=False, help="torch.compile the training passes")
@click.option("--eval-every", type=int, default=100, show_default=True, help="log the test loss every this many steps")
@click.option("--n-test", type=int, default=0, show_default=True, help="test examples of each test eval, a fresh random subset (0: the whole test set, always at the last step)")
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dir: Path, lrs: str, steps: str | None, runs: tuple[Path, ...], points: tuple[Path, ...], device: str, compile: bool, eval_every: int, n_test: int, show: bool):
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
    for lr, n_steps in zip(lr_list, steps_per_lr):
        run_dir = exp_dir / f"lr{lr:g}"
        if (run_dir / "full_model" / "metrics.pt").exists():
            run_steps[lr] = load_config(run_dir).n_steps
            click.echo(f"lr {lr:g}: reusing {run_dir} ({run_steps[lr]} steps)")
        else:
            run_dir.mkdir(exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps(asdict(replace(template, lr=lr, n_steps=n_steps)), indent=2))
            click.echo(f"lr {lr:g}: training {run_dir} for {n_steps} steps")
            with click.Context(train) as ctx:
                ctx.invoke(train, exp_dir=run_dir, device=device, compile=compile, eval_every=eval_every, n_test=n_test, show=False)
            run_steps[lr] = n_steps
        run_metrics[lr] = total_norms(torch.load(run_dir / "full_model" / "metrics.pt", weights_only=True))

    ema_gamma, ema_decay = 16.97, 0.9  # the defaults of `main.py train`
    acc_models = {"": "model", "/ema": f"EMA (gamma {ema_gamma:g})", "/ema_short": f"EMA (decay {ema_decay:g})"}
    acc_models = {suffix: label for suffix, label in acc_models.items() if all(f"test_acc{suffix}" in m for m in run_metrics.values())}
    losses = [run_metrics[lr]["test_loss"]["value"][-1].item() for lr in lr_list]
    max_accs = {suffix: [run_metrics[lr][f"test_acc{suffix}"]["value"].max().item() for lr in lr_list] for suffix in acc_models}
    min_losses = {suffix: [run_metrics[lr][f"test_loss{suffix}"]["value"].min().item() for lr in lr_list] for suffix in acc_models}
    max_acc_steps = {suffix: [int(run_metrics[lr][f"test_acc{suffix}"]["step"][run_metrics[lr][f"test_acc{suffix}"]["value"].argmax()]) for lr in lr_list]
                     for suffix in acc_models}
    best = lr_list[min(range(len(lr_list)), key=lambda i: losses[i])]
    for i, lr in enumerate(lr_list):
        click.echo(f"  lr {lr:g}: {run_steps[lr]} steps, test loss {losses[i]:.4f}, max acc " + " ".join(f"{acc_models[s]} {max_accs[s][i]:.3f} (step {max_acc_steps[s][i]})" for s in acc_models))
    steps_list = [run_steps[lr] for lr in lr_list]
    summary = {"steps": steps_list, "lrs": lr_list, "test_loss": losses, "best": best, "ema_gamma": ema_gamma, "ema_decay": ema_decay,
               **{f"max_test_acc{suffix.replace('/', '_')}": values for suffix, values in max_accs.items()},
               **{f"max_test_acc_step{suffix.replace('/', '_')}": values for suffix, values in max_acc_steps.items()},
               **{f"min_test_loss{suffix.replace('/', '_')}": values for suffix, values in min_losses.items()}}
    (exp_dir / "lr_tuning.json").write_text(json.dumps(summary, indent=2))
    torch.save({"train": run_metrics, "magic": {}}, exp_dir / "lr_tuning_runs.pt")

    steps_label = f"{steps_list[0]} steps" if len(set(steps_list)) == 1 else f"{min(steps_list)}-{max(steps_list)} steps"
    title = f"{template.model} on {template.dataset}, {template.optimizer}, {steps_label}"
    norm_names = {"params", "grads"}
    names = [n for n in run_metrics[lr_list[0]] if all(n in m for m in run_metrics.values())
             and n not in ("lr", "gpu_gb", "gpu_util") and ("/ema" not in n or n.startswith(("test_loss/", "test_acc/")))]
    names.sort(key=lambda n: (n in norm_names, n.startswith("test_")))
    plot_runs(run_metrics, names, title, exp_dir / "lr_tuning_curves.png", log_names=norm_names)
    fig = plot_summary(lr_list, min_losses, max_accs, max_acc_steps, acc_models, None, [None] * len(lr_list), best, title, exp_dir / "lr_tuning.png")
    for point, marker in zip(points, ("r*", "m*", "c*", "y*")):
        cfg, metrics = load_config(point), torch.load(point / "full_model" / "metrics.pt", weights_only=True)
        accs, done = metrics["test_acc"], (point / "full_model" / "final.pt").exists()
        substeps = round(int(accs["step"][-1]) / cfg.n_steps) if done else Experiment(point).substeps
        values = (metrics["test_loss"]["value"].min().item(), accs["value"].max().item(), int(accs["step"][accs["value"].argmax()]) // substeps)
        label = f"{cfg.optimizer} lr {cfg.lr:g}" + ("" if done else f" ({int(accs['step'][-1]) // substeps} steps so far)")
        click.echo(f"  {label}: min test loss {values[0]:.4f}, max test acc {values[1]:.3f} at step {values[2]}")
        for ax, value in zip(fig.axes, values):
            ax.plot([cfg.lr], [value], marker, ms=12, label=label)
            ax.legend(fontsize=7, frameon=False)
    if points:
        fig.savefig(exp_dir / "lr_tuning.png", dpi=150)
    click.echo(f"saved {exp_dir / 'lr_tuning_curves.png'} and {exp_dir / 'lr_tuning.png'}")
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
