"""
MAGIC backward of an lr sweep (the run dirs DIR/lr<lr> of experiments.lr_sweep) from step --end to step --start, to see how much
df/ds_t and df/dw_t grow over it:
  python -m experiments.backprop_sweep DIR [--lrs 0.01,0.1] [--start -1000] [--end 0] [--n-test 1] [--subset 0] [--device cuda] [--no-show]
--start / --end: steps of the training, counted from n_steps if <= 0 (default: n_steps - 1000 and n_steps, so each lr its own); each
must have a saved state (`main.py train --checkpoint-every`, final.pt at n_steps). f = the mean loss of the --subset-th random set
of --n-test test examples at step --end. Runs whose DIR/lr<lr>/backprop/<spec>/metrics.pt exists are reused (default --lrs: every
DIR/lr<lr> with a trained model). Saves, over every lr with a result:
- DIR/backprop_tuning.png (+ .json): the growth over the backward vs lr, one line per (start, end) spec found:
  |df/ds_start| / |df/ds_end| and |df/dw_start| / |df/dw_{end-1}| (from the earliest finite step if the backward overflowed, marked)
- DIR/backprop_curves_<spec>.png: f(s_t), |df/ds_t| and |df/dw_t|, raw and divided by their value at the start of the backward,
  vs t - end, one line per lr
"""

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from experiments.lr_sweep import load_config
from experiments.tune_lr import Metrics, d_state_totals, plot_runs
from main import Experiment
from src.logging import Logger, fig_title
from src.logging import total_l2
from src.magic import backprop_logged, to_device
from src.optimizers import OptimState


def spec_name(start: int, end: int) -> str:
    """The dir name of a (start, end) spec, e.g. "n-1000_to_n" (n = n_steps)"""
    return "_to_".join(str(v) if v > 0 else "n" if v == 0 else f"n{v}" for v in (start, end))


def run_steps(cfg_steps: int, start: int, end: int) -> tuple[int, int]:
    """(start, end) of a spec as steps of a run with `cfg_steps` steps"""
    return tuple(v if v > 0 else cfg_steps + v for v in (start, end))


def saved_state(exp: Experiment, step: int) -> OptimState:
    """The state of the full model's training at the optimizer step `step` (final.pt, a checkpoint or the start state)"""
    first, n_steps = exp.steps
    if step == first + n_steps:
        return exp.load_state(exp.dir / "full_model" / "final.pt")
    if step == first:
        return exp.start_state
    checkpoints = exp.checkpoints()
    if step not in checkpoints:
        raise click.UsageError(f"no saved state at step {step} of {exp.dir} (checkpoints: {list(checkpoints)}): train it with --checkpoint-every")
    return exp.load_state(checkpoints[step])


def backprop_run(run_dir: Path, out_dir: Path, start: int, end: int, n_test: int, subset: int, device: str, compile: bool, passes: int):
    """Backpropagate f from step `end` to `start` of the run, logging f(s_t), |df/ds_t| (also at `end`) and |df/dw_t| into out_dir/metrics.pt"""
    exp = Experiment(run_dir, device, compile)
    t_start, t_end = (t * exp.substeps for t in run_steps(exp.cfg.n_steps, start, end))
    assert t_start < t_end, f"start {t_start} >= end {t_end}"
    eval_set = exp.test_ds.examples(exp.test_ds.subset(n_test, index=subset))
    end_state = saved_state(exp, t_end)
    f, f_grads = exp.objective.eval_grads(to_device(end_state.params, exp.device), [eval_set])
    d_state = end_state.cotangent(f_grads)
    logger = Logger(f"backprop {t_end} -> {t_start}; {exp.name}", smooth=False, lags=(), step_scale=exp.substeps)
    logger.log(t_end, **{"f/0": f[0], "d_state/0": total_l2(d_state, 1)[0]})
    backprop_logged(exp.magic, saved_state(exp, t_start), d_state, [eval_set], logger, n_steps=t_end - t_start, start=t_start, passes=passes)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.finish(out_dir / "metrics.pt")


def growth(steps: Tensor, values: Tensor) -> tuple[float, int | None]:
    """(value at the earliest step with a finite value / value at the last step, the last step that overflowed or None)"""
    order = steps.argsort()
    steps, values = steps[order], values[order]
    finite = values.isfinite()
    overflow = None if finite.all() else int(steps[~finite].max())
    return (values[finite][0] / values[-1]).item(), overflow


def curves(metrics: Metrics, t_end: int) -> Metrics:
    """The plotted curves of a backward, against t - end"""
    steps, d_state = d_state_totals(metrics)
    d_weights = metrics["d_weights/0"]
    order = d_weights["step"].argsort()
    w_steps, w_values = d_weights["step"][order], d_weights["value"][order]
    return {
        "f(s_t)": {"step": metrics["f/0"]["step"] - t_end, "value": metrics["f/0"]["value"]},
        "|df/ds_t|": {"step": steps - t_end, "value": d_state},
        "|df/ds_t| / |df/ds_end|": {"step": steps - t_end, "value": d_state / d_state[-1]},
        "|df/dw_t|": {"step": w_steps - t_end, "value": w_values},
        "|df/dw_t| / |df/dw_(end-1)|": {"step": w_steps - t_end, "value": w_values / w_values[-1]},
    }


def plot_growth(results: dict[str, dict], title: str, path: Path):
    """backprop_tuning.png: the two growth ratios vs lr, one line per spec (overflowed backwards marked)"""
    panels = [("d_state", "|df/ds_start| / |df/ds_end|"), ("d_weights", "|df/dw_start| / |df/dw_(end-1)|")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.8), layout="constrained")
    all_lrs = sorted({lr for r in results.values() for lr in r["lrs"]})
    for ax, (key, label) in zip(axes, panels):
        for spec, r in results.items():
            ax.plot(r["lrs"], r[f"{key}_growth"], "o-", label=spec.replace("_to_", " -> "))
            overflowed = [(lr, g) for lr, g, t in zip(r["lrs"], r[f"{key}_growth"], r["overflow_step"]) if t is not None]
            if overflowed:
                ax.plot(*zip(*overflowed), "rx", ms=8)
        ax.axhline(1, color="gray", ls="--", lw=0.8)
        ax.set(xscale="log", yscale="log", xlabel="lr", ylabel=label)
        ax.set_xticks(all_lrs, [f"{lr:g}" for lr in all_lrs], rotation=45)
        ax.tick_params(which="minor", labelbottom=False)
        ax.legend(title="backprop (n = n_steps)", fontsize=7, title_fontsize=7, frameon=False)
    fig_title(fig, title + "\nred x: |df/ds_t| overflowed (ratio from its earliest finite step)")
    fig.savefig(path, dpi=150)


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option("--lrs", type=str, default=None, help="comma-separated learning rates to backprop (default: every DIR/lr<lr> with final.pt)")
@click.option("--start", type=int, default=-1000, show_default=True, help="backprop down to this step (<= 0: from n_steps)")
@click.option("--end", type=int, default=0, show_default=True, help="backprop from this step (<= 0: from n_steps)")
@click.option("--n-test", type=int, default=1, show_default=True, help="test examples of the eval set")
@click.option("--subset", type=int, default=0, show_default=True, help="index of the eval set (`Dataset.subset`)")
@click.option("--passes", type=int, default=2, show_default=True, help="see `main.py magic --passes`")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--compile/--no-compile", default=False, help="torch.compile the training passes")
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dir: Path, lrs: str | None, start: int, end: int, n_test: int, subset: int, passes: int, device: str, compile: bool, show: bool):
    run_dirs = {load_config(d).lr: d for d in exp_dir.glob("lr*") if (d / "config.json").exists()}
    trained = [lr for lr, d in run_dirs.items() if (d / "full_model" / "final.pt").exists()]
    todo = sorted(trained if lrs is None else [float(lr) for lr in lrs.split(",")])
    eval_name = f"{n_test}ex{subset}"
    for lr in todo:
        if lr not in run_dirs:
            raise click.UsageError(f"no run dir of lr {lr:g} in {exp_dir}")
        out_dir = run_dirs[lr] / "backprop" / eval_name / spec_name(start, end)
        if (out_dir / "metrics.pt").exists():
            click.echo(f"lr {lr:g}: reusing {out_dir}")
        else:
            click.echo(f"lr {lr:g}: backprop into {out_dir}")
            backprop_run(run_dirs[lr], out_dir, start, end, n_test, subset, device, compile, passes)

    template = load_config(exp_dir)
    title = f"{template.model} on {template.dataset}, {template.optimizer}; f = loss on {n_test} test examples (set {subset})"
    results: dict[str, dict] = {}
    for spec in sorted({p.parent.name for d in run_dirs.values() for p in (d / "backprop" / eval_name).glob("*/metrics.pt")}):
        runs: dict[float, Metrics] = {}
        result = {"lrs": [], "steps": [], "d_state_growth": [], "d_weights_growth": [], "overflow_step": []}
        for lr in sorted(run_dirs):
            path = run_dirs[lr] / "backprop" / eval_name / spec / "metrics.pt"
            if not path.exists():
                continue
            metrics = torch.load(path, weights_only=True)
            steps, d_state = d_state_totals(metrics)
            d_weights = metrics["d_weights/0"]
            (state_growth, overflow), (weights_growth, _) = growth(steps, d_state), growth(d_weights["step"], d_weights["value"])
            t_end = int(steps.max())
            runs[lr] = curves(metrics, t_end)
            for key, value in zip(result, (lr, (int(steps.min()), t_end), state_growth, weights_growth, overflow)):
                result[key].append(value)
            click.echo(f"  {spec} lr {lr:g} (steps {int(steps.min())} -> {t_end}): |df/ds| x{state_growth:.3g}, |df/dw| x{weights_growth:.3g}"
                       + ("" if overflow is None else f" (overflowed at step {overflow})"))
        results[spec] = result
        names = list(next(iter(runs.values())))
        plot_runs(runs, names, f"backprop {spec.replace('_to_', ' -> ')} (n = n_steps); {title}", exp_dir / f"backprop_curves_{spec}.png",
                  log_names=set(names) - {"f(s_t)"}, smooth=False, xlabel="t - end")
    if results:
        (exp_dir / "backprop_tuning.json").write_text(json.dumps(results, indent=2))
        plot_growth(results, title, exp_dir / "backprop_tuning.png")
        click.echo(f"saved {exp_dir / 'backprop_tuning.png'} and the curves {exp_dir}/backprop_curves_*.png")
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
