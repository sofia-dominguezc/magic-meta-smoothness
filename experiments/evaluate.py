"""

Compare MAGIC's linear predictions (DIR/magic) with the retrained models (DIR/retrained) of an experiment:
  python -m experiments.evaluate DIR [--device cpu] [--no-show] [--variant V]
--variant: the predictions of `main.py magic --variant V` (DIR/magic_V), into DIR/evaluation_V.

Every data point is (predicted, real) change of f = the loss on one eval subset, for one retrained model, where
predicted = betas @ (w - 1) (MAGIC's linearization) and real = f(retrained model) - f(DIR/full_model/final.pt, trained
vmapped like the retrained models, see main.py). The models are grouped by their perturbation (kind, alpha). Outputs,
in DIR/evaluation:

- <kind>_<alpha>.png: one panel per eval subset (a square-ish block of panels per eval subset size), one point per
  retrained model of that (kind, alpha), with the agreement metrics of each panel
- <kind>_<alpha>_pooled.png: all those points in one big scatter plot (colored by eval subset), with the pooled ccc
  and spearman
- results.json: per (kind, alpha), the metrics per eval subset, and "pooled": the metrics over all its data points
  (all eval subsets and retrained models together)
- alpha.png (if there are several (kind, alpha)): the pooled ccc and spearman vs. alpha, one line per kind, with the
  per-eval-subset values as faint points
- <kind>_<alpha>_reflected.png, <kind>_<alpha>_reflected_pooled.png, alpha_reflected.png, and "reflected" in results.json:
  the same, with MAGIC's prediction on the x axis replaced by the real change measured at 2 ema - params (the params
  reflected through their EMA `ema_params`, the only place it is used: one oscillation off at the edge of stability),
  a reference for how much of the disagreement with MAGIC is the noise of the final params themselves.
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

from main import Experiment, HESSIAN, magic_dirname
from src.logging import fig_title
from src.utils import dict_zip


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
            ax.plot([g["alpha"] for g in gs], [g["pooled"][metric] for g in gs], marker=marker, color=color, lw=1.2, label=kind, zorder=2)
        ax.axhline(0.9, color="gray", ls="--", lw=0.8, zorder=0)
        ax.set(xscale="log", xlabel="alpha", ylabel=f"pooled {metric}", ylim=(min(0.5, ax.get_ylim()[0]), 1.05))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig_title(fig, title)
    fig.savefig(out_path, dpi=150)


def evaluate(exp_dir: Path, device: str = "cpu", show: bool = True, variant: str = HESSIAN) -> dict:
    """`variant`: evaluate DIR/magic_<variant> (the predictions of `main.py magic --variant`) into DIR/evaluation_<variant>"""
    exp = Experiment(exp_dir, device)
    cfg = exp.cfg
    magic_dir, method = exp_dir / magic_dirname(variant), "MAGIC" if variant == HESSIAN else f"MAGIC ({variant})"
    if not (exp_dir / "full_model" / "final.pt").exists():
        raise click.UsageError(f"no full model {exp_dir / 'full_model' / 'final.pt'}: run `python main.py train`")
    lins = {d: torch.load(d / "linearization.pt", weights_only=True)
            for d in magic_dir.glob("*") if (d / "linearization.pt").exists()}  # dir -> {indices, output, beta}
    lins = dict(sorted(lins.items(), key=lambda item: (len(item[1]["indices"]), natural(item[0].name))))
    if not lins:
        raise click.UsageError(f"no eval sets in {magic_dir}: run `python main.py magic{'' if variant == HESSIAN else f' --variant {variant}'}`")
    eval_sets = [l["indices"] for l in lins.values()]
    outputs, betas = torch.stack([l["output"] for l in lins.values()]), torch.stack([l["beta"] for l in lins.values()])  # [n_fns], [n_fns, N]
    runs = {d: torch.load(d / "perturbation.pt", weights_only=True)
            for d in sorted((exp_dir / "retrained").glob("*"), key=lambda d: natural(d.name)) if (d / "final.pt").exists()}  # dir -> {weights, kind, alpha}
    if not runs:
        raise click.UsageError(f"no retrained models in {exp_dir / 'retrained'}: run `python main.py retrain`")
    click.echo(f"evaluating {len(lins)} eval sets on {len(runs)} retrained models")

    def f(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """The eval fns on the state saved at `path`: at its params, and at 2 ema - params."""
        state = exp.load_state(path)
        reflected = {k: 2 * e - p for k, (e, p) in dict_zip(state.ema_params, state.params).items()}
        return tuple(
            torch.stack([exp.magic.eval_loss({k: t.to(device) for k, t in params.items()}, exp.test_loader, indices)[0]
                         for indices in eval_sets]).cpu()
            for params in (state.params, reflected)
        )

    preds, reals, ema_reals = [], [], []  # [n_fns] per run: the predicted and real change of the eval fns (and at the EMA)
    f_full, f_full_ema = f(exp_dir / "full_model" / "final.pt")
    for run_dir, p in tqdm(runs.items(), desc="retrained models"):
        preds.append(betas @ (p["weights"] - 1).to(betas.dtype))  # f(w) - f(1) ~ betas @ (w - 1)
        f_run, f_run_ema = f(run_dir / "final.pt")
        reals.append(f_run - f_full)
        ema_reals.append(f_run_ema - f_full_ema)
    preds, reals, ema_reals = (torch.stack(x, 1).numpy() for x in (preds, reals, ema_reals))  # [n_fns, n_runs]
    full, full_ema = outputs.numpy(), f_full_ema.numpy()
    perts = [(p["kind"], p["alpha"]) for p in runs.values()]
    ema_label, ema_method = "real change at 2 ema - params", "params reflected through their EMA"

    out_dir = exp_dir / ("evaluation" if variant == HESSIAN else f"evaluation_{variant}")
    out_dir.mkdir(exist_ok=True)
    sizes = [len(indices) for indices in eval_sets]
    rows = list(dict.fromkeys(sizes))  # one square-ish block of rows per eval set size
    ncols = math.ceil(math.sqrt(max(map(sizes.count, rows))))
    first_row = dict(zip(rows, np.cumsum([0] + [-(-sizes.count(n) // ncols) for n in rows]).tolist()))
    ranks = [sizes[:i].count(n) for i, n in enumerate(sizes)]  # among the sets of its size
    positions = [(first_row[n] + k // ncols, k % ncols) for n, k in zip(sizes, ranks)]
    titles = [f"loss on {d.name}" for d in lins]
    results = {"eval_sets": [{"dir": d.name, "n": n} for d, n in zip(lins, sizes)], "full": full.tolist(), "full_reflected": full_ema.tolist(), "groups": []}
    for kind, alpha in sorted(set(perts)):
        sel = np.array([p == (kind, alpha) for p in perts])
        name = f"{kind}_{alpha:g}"
        title = f"{method} vs. retraining with {kind} {alpha:g} from step {cfg.start}: {sel.sum()} models\n{exp.name}"
        metrics = plot_scatter_grid(preds[:, sel], reals[:, sel], full, positions, titles,
                                    [f"{kind} {alpha:g}"] * int(sel.sum()), out_dir / f"{name}.png", title)
        pooled = plot_pooled_scatter(preds[:, sel], reals[:, sel], out_dir / f"{name}_pooled.png",
                                     f"{method} vs. retraining with {kind} {alpha:g} from step {cfg.start}: all {sel.sum()} models x {len(lins)} eval subsets\n{exp.name}")
        ema_metrics = plot_scatter_grid(ema_reals[:, sel], reals[:, sel], full_ema, positions, titles,
                                        [f"{kind} {alpha:g}"] * int(sel.sum()), out_dir / f"{name}_reflected.png",
                                        title.replace(method, ema_method, 1), xlabel=ema_label)
        ema_pooled = plot_pooled_scatter(ema_reals[:, sel], reals[:, sel], out_dir / f"{name}_reflected_pooled.png",
                                         f"{ema_method} vs. retraining with {kind} {alpha:g} from step {cfg.start}: all {sel.sum()} models x {len(lins)} eval subsets\n{exp.name}",
                                         xlabel=ema_label)
        results["groups"].append({
            "kind": kind, "alpha": alpha, "runs": [d.name for d, p in zip(runs, perts) if p == (kind, alpha)],
            "metrics": metrics, "pooled": pooled, "preds": preds[:, sel].tolist(), "reals": reals[:, sel].tolist(),
            "reflected": {"metrics": ema_metrics, "pooled": ema_pooled, "reals": ema_reals[:, sel].tolist()},
        })
        for what, ms, pl in (("", metrics, pooled), (" (reflected reference)", ema_metrics, ema_pooled)):
            click.echo(f"{kind} {alpha:g}{what}:")
            for d, m in zip(lins, ms):
                click.echo(f"  {d.name}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
            click.echo(f"  pooled: " + ", ".join(f"{k}={v:.3f}" for k, v in pl.items()))
    if len(results["groups"]) > 1:
        plot_alpha(results["groups"], out_dir / "alpha.png", f"{method} vs. retraining from step {cfg.start}, pooled over all eval subsets and models\n{exp.name}")
        plot_alpha([{**g, **g["reflected"]} for g in results["groups"]], out_dir / "alpha_reflected.png",
                   f"{ema_method} vs. retraining from step {cfg.start}, pooled over all eval subsets and models\n{exp.name}")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    if show:
        plt.show()
    plt.close("all")
    return results


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--show/--no-show", default=True, help="show the plots")
@click.option("--variant", type=str, default=HESSIAN, show_default=True, help="evaluate DIR/magic_<variant> into DIR/evaluation_<variant>")
def main(exp_dir, device, show, variant):
    evaluate(exp_dir, device, show, variant)


if __name__ == "__main__":
    main()
