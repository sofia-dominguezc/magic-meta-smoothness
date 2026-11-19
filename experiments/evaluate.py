"""

Compare MAGIC's linear predictions (DIR/magic) with the retrained models (DIR/retrained) of an experiment:
  python -m experiments.evaluate DIR [--device cpu] [--no-show] [--extra-step]

Every data point is (predicted, real) change of f = the loss on one eval subset, for one retrained model, where
predicted = d_weights @ (w - 1) (MAGIC's linearization) and real = f(retrained model) - f(DIR/full_model/final.pt). The losses are read from the models' losses.json where saved by an earlier
evaluation (`Experiment.eval_losses`), so pruned experiments (`main.py prune`) can be re-evaluated; the full model's
f falls back to the one saved by `magic`. The models are grouped by their perturbation (kind, alpha).
Outputs, in DIR/evaluation, with <group> = <kind>_<alpha>:

- <group>.png: one panel per eval subset (a square-ish block of panels per eval subset size), one point per
  retrained model of that group, with the agreement metrics of each panel
- <group>_pooled.png: all those points in one big scatter plot (colored by eval subset), with the pooled ccc
  and spearman
- results.json: per group, the metrics per eval subset, and "pooled": the metrics over all its data points
  (all eval subsets and retrained models together); "magic": per predictions dir, its eval subsets and f on the full model
- alpha.png (if there are several groups): the pooled ccc and spearman vs. alpha, one line per kind, with the
  per-eval-subset values as faint points
- with --extra-step: <group>_extra_step.png, <group>_extra_step_pooled.png, alpha_extra_step.png, and "extra_step" in results.json:
  the same, with MAGIC's prediction on the x axis replaced by the real change measured after one more training step of every
  model (on all the data), a reference for how much of the disagreement with MAGIC is the noise of the final params themselves.
"""

import json
import math
import re
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from main import Experiment, has_model, load_renamed
from src.logging import fig_title


def natural(name: str) -> list:
    """Sort key with the numbers in `name` compared as numbers ("5_examples_2" < "5_examples_10")."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def agreement_metrics(pred: np.ndarray, real: np.ndarray) -> dict[str, float]:
    """How well `pred` matches `real` under the assumption real = pred (slope 1, no offset).
    All are symmetric in (pred, real) and independent of their number.
    - ccc:       Lin's concordance correlation, 2 cov / (var_p + var_r + (mean_p - mean_r)^2) = 1 - E(p - r)^2 / (...):
                 the R^2 of the identity line; in [-1, 1], 1 iff pred = real; penalizes slope and offset errors
    - spearman:  Spearman rank correlation (ranking agreement only; insensitive to slope and offset)
    - tls_slope: total-least-squares slope through the origin (1 is perfect; symmetric: slope(r, p) = 1/slope(p, r))
    """
    p, r = pred.astype(np.float64), real.astype(np.float64)
    eps = 1e-12
    cov = np.mean((p - p.mean()) * (r - r.mean()))
    d = p @ p - r @ r
    tls_slope = (d + np.sqrt(d**2 + 4 * (r @ p) ** 2)) / (2 * (r @ p) + eps)
    return {
        "ccc": float(2 * cov / (p.var() + r.var() + (p.mean() - r.mean()) ** 2 + eps)),
        "spearman": float(np.corrcoef(p.argsort().argsort(), r.argsort().argsort())[0, 1]) if len(p) > 1 else float("nan"),  # (no ties: floats)
        "tls_slope": float(min(tls_slope, 1 / tls_slope)),
    }


SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]  # second encoding: past 3 colors, hues alone aren't distinguishable


def plot_scatter_grid(
    preds: np.ndarray,
    reals: np.ndarray,
    full: np.ndarray,
    positions: list[tuple[int, int]],
    titles: list[str],
    groups: list[str],
    out_path: Path,
    title: str = "",
    xlabel: str = "predicted change of the loss (MAGIC)",
) -> list[dict[str, float]]:
    """One panel per eval fn i, at grid cell positions[i] = (row, col), with one point per run.
    preds, reals: [n_fns, n_runs] predicted and real changes of each eval fn on each run;
    full: [n_fns] its value on the full model (shown); groups: [n_runs] what colors each point; `xlabel`: what preds are.
    Returns the agreement metrics of (pred, real) per eval fn."""
    nrows, ncols = max(r for r, _ in positions) + 1, max(c for _, c in positions) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), squeeze=False, layout="constrained")
    used = set(positions)
    for (r, c), ax in np.ndenumerate(axes):
        if (r, c) not in used:
            fig.delaxes(ax)
    categories = list(dict.fromkeys(groups))
    group_idx = np.array([categories.index(g) for g in groups])
    all_metrics = []
    for i, (r, c) in enumerate(positions):
        ax, x, y, f = axes[r, c], preds[i], reals[i], full[i]
        m = agreement_metrics(x, y)
        all_metrics.append(m)
        lo, hi = min(x.min(), y.min(), 0), max(x.max(), y.max(), 0)
        pad = 0.05 * (hi - lo + 1e-12)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="#52514e", lw=1, ls="--", zorder=1)
        ax.axhline(0, color="#a3a29b", lw=0.8, zorder=0)
        ax.axvline(0, color="#a3a29b", lw=0.8, zorder=0)
        for k, cat in enumerate(categories):
            sel = group_idx == k
            ax.scatter(
                x[sel], y[sel], s=30, color=SERIES_COLORS[k % len(SERIES_COLORS)],
                marker=SERIES_MARKERS[k % len(SERIES_MARKERS)], edgecolors="white", linewidths=0.6,
                label=cat, zorder=2,
            )
        info = f"full-data loss: {f:.4f}\n" + "\n".join(f"{k}: {v:.3f}" for k, v in m.items())
        ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=8, va="top", color="#0b0b0b",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#d6d5cf", alpha=0.9))
        ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
        ax.set_title(titles[i], fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("real change (retrained - full model)", fontsize=9)
        ax.tick_params(labelsize=8, colors="#52514e")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[positions[0]].get_legend_handles_labels()
    fig_title(fig, title, fontsize=12)
    fig.legend(handles, labels, title="retrained models", loc="outside lower center", ncols=min(len(labels), 6),
               fontsize=8, title_fontsize=8, frameon=False)
    fig.savefig(out_path, dpi=150)
    return all_metrics


def plot_pooled_scatter(
    preds: np.ndarray, reals: np.ndarray, out_path: Path, title: str = "", xlabel: str = "predicted change of the loss (MAGIC)"
) -> dict[str, float]:
    """All the points of a (kind, alpha) in one plot: preds, reals [n_fns, n_runs], one color per eval fn (no legend),
    with the pooled ccc and spearman in large print. Returns the pooled metrics."""
    pooled = agreement_metrics(preds.ravel(), reals.ravel())
    fig, ax = plt.subplots(figsize=(9, 9), layout="constrained")
    lo, hi = min(preds.min(), reals.min(), 0), max(preds.max(), reals.max(), 0)
    pad = 0.05 * (hi - lo + 1e-12)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color="#52514e", lw=1.2, ls="--", zorder=1)
    ax.axhline(0, color="#a3a29b", lw=0.8, zorder=0)
    ax.axvline(0, color="#a3a29b", lw=0.8, zorder=0)
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(preds)))
    for i in range(len(preds)):
        ax.scatter(preds[i], reals[i], s=36, color=colors[i], edgecolors="white", linewidths=0.5, alpha=0.85, zorder=2)
    ax.text(0.03, 0.97, f"ccc = {pooled['ccc']:.3f}\nspearman = {pooled['spearman']:.3f}", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top", color="#0b0b0b",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d6d5cf", alpha=0.9))
    ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("real change (retrained - full model)", fontsize=13)
    ax.tick_params(labelsize=11, colors="#52514e")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig_title(fig, title, fontsize=13)
    fig.savefig(out_path, dpi=150)
    return pooled


def plot_alpha(groups: list[dict], out_path: Path, title: str = "", metrics: tuple[str, ...] = ("ccc", "spearman")):
    """The pooled `metrics` of each (kind, alpha) group (results.json "groups") vs. alpha, one panel per metric,
    one line per kind, with the per-eval-subset values as faint points."""
    kinds = list(dict.fromkeys(g["kind"] for g in groups))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 3.6), squeeze=False, layout="constrained")
    for ax, metric in zip(axes[0], metrics):
        for k, kind in enumerate(kinds):
            gs = sorted((g for g in groups if g["kind"] == kind), key=lambda g: g["alpha"])
            color, marker = SERIES_COLORS[k % len(SERIES_COLORS)], SERIES_MARKERS[k % len(SERIES_MARKERS)]
            for g in gs:
                ax.scatter([g["alpha"]] * len(g["metrics"]), [m[metric] for m in g["metrics"]], s=12, color=color, alpha=0.3, zorder=1)
            ax.plot([g["alpha"] for g in gs], [g["pooled"][metric] for g in gs], marker=marker, color=color, lw=1.2,
                    label=kind, zorder=2)
        ax.axhline(0.9, color="gray", ls="--", lw=0.8, zorder=0)
        ax.set(xscale="log", xlabel="alpha", ylabel=f"pooled {metric}", ylim=(min(0.5, ax.get_ylim()[0]), 1.05))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig_title(fig, title)
    fig.savefig(out_path, dpi=150)


def evaluate(exp_dir: Path, device: str = "cpu", show: bool = True, extra_step: bool = False) -> dict:
    """`extra_step`: also the reference of the real changes after one more training
    step of every model."""
    exp = Experiment(exp_dir, device)
    cfg = exp.cfg
    runs = {d: torch.load(d / "perturbation.pt", weights_only=True)
            for d in sorted((exp_dir / "retrained").glob("*"), key=lambda d: natural(d.name)) if has_model(d)}  # dir -> {weights, kind, alpha}
    if not runs:
        raise click.UsageError(f"no retrained models in {exp_dir / 'retrained'}: run `python main.py retrain`")
    out_dir = exp_dir / "evaluation"
    out_dir.mkdir(exist_ok=True)
    results: dict = {"magic": {}, "groups": []}
    extra_label, extra_method = "real change after one more step", "the models one step further"
    magic_dir = exp_dir / "magic"
    lins = {d: load_renamed(d / "linearization.pt")
            for d in magic_dir.glob("*") if (d / "linearization.pt").exists()}  # dir -> {indices, output, d_weights}
    lins = dict(sorted(lins.items(), key=lambda item: (len(item[1]["indices"]), natural(item[0].name))))
    if not lins:
        raise click.UsageError(f"no eval sets in {magic_dir}: run `python main.py magic`")
    eval_sets = [l["indices"] for l in lins.values()]
    outputs, d_weights = torch.stack([l["output"] for l in lins.values()]), torch.stack([l["d_weights"] for l in lins.values()])  # [n_fns], [n_fns, N]
    click.echo(f"evaluating {len(lins)} eval sets on {len(runs)} retrained models")
    steps = [False, True] if extra_step else [False]
    try:
        f_full = [exp.eval_losses(exp_dir / "full_model", eval_sets, extra_step=e) for e in steps]  # [n_fns], and after the extra step
    except click.UsageError as e:
        if extra_step:
            raise
        click.echo(f"{e.message}: using the full model's f saved by magic")
        f_full = [outputs]
    preds = torch.stack([d_weights @ (p["weights"] - 1).to(d_weights.dtype) for p in runs.values()], 1).numpy()  # f(w) - f(1) ~ d_weights @ (w - 1); [n_fns, n_runs]
    reals, extra_reals = (torch.stack([exp.eval_losses(d, eval_sets, extra_step=e) - f_full[i] for d in tqdm(runs, desc="retrained models")], 1).numpy()
                          if i < len(f_full) else None for i, e in enumerate((False, True)))
    full = outputs.numpy()

    sizes = [len(indices) for indices in eval_sets]
    rows = list(dict.fromkeys(sizes))  # one square-ish block of rows per eval set size
    ncols = math.ceil(math.sqrt(max(map(sizes.count, rows))))
    first_row = dict(zip(rows, np.cumsum([0] + [-(-sizes.count(n) // ncols) for n in rows]).tolist()))
    ranks = [sizes[:i].count(n) for i, n in enumerate(sizes)]  # among the sets of its size
    positions = [(first_row[n] + k // ncols, k % ncols) for n, k in zip(sizes, ranks)]
    titles = [f"loss on {d.name}" for d in lins]
    results["magic"][magic_dir.name] = {"eval_sets": [{"dir": d.name, "n": n} for d, n in zip(lins, sizes)], "full": full.tolist(),
                                        **({"full_extra_step": f_full[1].numpy().tolist()} if extra_step else {})}
    perts = [(p["kind"], p["alpha"]) for p in runs.values()]
    for kind, alpha in sorted(set(perts)):
        sel = np.array([p == (kind, alpha) for p in perts])
        name = f"{kind}_{alpha:g}"
        label = f"{kind} {alpha:g}"
        title = f"MAGIC vs. retraining with {label} from step {cfg.start}: {sel.sum()} models\n{exp.name}"
        metrics = plot_scatter_grid(preds[:, sel], reals[:, sel], full, positions, titles, [label] * int(sel.sum()), out_dir / f"{name}.png", title)
        pooled = plot_pooled_scatter(preds[:, sel], reals[:, sel], out_dir / f"{name}_pooled.png",
                                     f"MAGIC vs. retraining with {label} from step {cfg.start}: all {sel.sum()} models x {len(lins)} eval subsets\n{exp.name}")
        group = {
            "kind": kind, "alpha": alpha, "magic": magic_dir.name,
            "runs": [d.name for d, p in zip(runs, perts) if p == (kind, alpha)],
            "metrics": metrics, "pooled": pooled, "preds": preds[:, sel].tolist(), "reals": reals[:, sel].tolist(),
        }
        references = [("", metrics, pooled)]
        if extra_reals is not None:
            extra_metrics = plot_scatter_grid(extra_reals[:, sel], reals[:, sel], f_full[1].numpy(), positions, titles, [label] * int(sel.sum()),
                                              out_dir / f"{name}_extra_step.png", title.replace("MAGIC", extra_method, 1), xlabel=extra_label)
            extra_pooled = plot_pooled_scatter(extra_reals[:, sel], reals[:, sel], out_dir / f"{name}_extra_step_pooled.png",
                                               f"{extra_method} vs. retraining with {label} from step {cfg.start}: all {sel.sum()} models x {len(lins)} eval subsets\n{exp.name}",
                                               xlabel=extra_label)
            group["extra_step"] = {"metrics": extra_metrics, "pooled": extra_pooled, "reals": extra_reals[:, sel].tolist()}
            references.append((" (extra step reference)", extra_metrics, extra_pooled))
        results["groups"].append(group)
        for what, ms, pl in references:
            click.echo(f"{label}{what}:")
            for d, m in zip(lins, ms):
                click.echo(f"  {d.name}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
            click.echo(f"  pooled: " + ", ".join(f"{k}={v:.3f}" for k, v in pl.items()))
    if len(results["groups"]) > 1:
        plot_alpha(results["groups"], out_dir / "alpha.png", f"MAGIC vs. retraining from step {cfg.start}, pooled over all eval subsets and models\n{exp.name}")
        if extra_step:
            plot_alpha([{**g, **g["extra_step"]} for g in results["groups"]], out_dir / "alpha_extra_step.png",
                       f"{extra_method} vs. retraining from step {cfg.start}, pooled over all eval subsets and models\n{exp.name}")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    if show:
        plt.show()
    plt.close("all")
    return results


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--show/--no-show", default=True, help="show the plots")
@click.option("--extra-step/--no-extra-step", default=False, help="also the reference of the real changes after one more training step of every model")
def main(exp_dir, device, show, extra_step):
    evaluate(exp_dir, device, show, extra_step)


if __name__ == "__main__":
    main()
