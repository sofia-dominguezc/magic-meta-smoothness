"""
Which batch size, at a fixed number of steps? For each batch size, tune the lr on a half-width model (μP: the lr transfers
across widths; all the lrs of a batch size train at once, vmapped), then train the real model at the best lr:
  python -m experiments.batch_study run OUT --model cnn --optimizer adam [--steps 10000] [--device cuda]
  python -m experiments.batch_study plot OUT

`run` appends to OUT/<model>_<optimizer>.jsonl (and skips what is already there). Plots:
- lr_tuning.png: test loss of the EMA of the half-width models vs lr, per batch size
- batch_size.png: the best lr, and the test loss / acc of the real model there, vs the batch size
"""

import json
import time
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from experiments.arch_search import EMA_GAMMA, append_jsonl, build_model, read_jsonl, sync
from src.data import DATASETS, Loader
from src.objective import LOSSES, Objective, metrics
from src.logging import fig_title
from src.magic import Magic
from src.optimizers import AdamState, LRSchedule, SGDState
from src.utils import dict_zip

MODELS = {"cnn": (32, 2), "vit": (128, 4)}  # model -> (width, depth); the lr is tuned at half the width
BATCH_SIZES = [16, 32, 64, 128]
N_LRS = 10
LR_RATIO = 2**0.5  # between consecutive lrs
CENTER_LRS = {("cnn", "sgd"): 0.5, ("vit", "sgd"): 0.15, ("cnn", "adam"): 0.07, ("vit", "adam"): 0.05}  # at B = 128; ~ sqrt(B)


def lr_grid(model: str, optimizer: str, batch_size: int) -> list[float]:
    center = CENTER_LRS[model, optimizer] * (batch_size / 128) ** 0.5
    return [float(f"{center * LR_RATIO ** (i - (N_LRS - 1) / 2):.3g}") for i in range(N_LRS)]


def evaluate(magic: Magic, params: dict[str, Tensor], loader: Loader, indices: Tensor, batched: bool) -> tuple[Tensor, Tensor]:
    """Mean (loss, acc) of `params` on the examples `indices`; one per model [R] if `batched`"""
    forward = torch.func.vmap(magic.objective.forward, in_dims=(0, None, None)) if batched else magic.objective.forward
    loss, acc = 0.0, 0.0
    with torch.no_grad():
        for X, Y in loader.iter_batches(indices):
            batch_loss, out = forward(params, X, Y)
            batch_acc = (out.argmax(-1) == Y).to(out.dtype).mean(-1)
            loss, acc = loss + batch_loss * len(X) / len(indices), acc + batch_acc * len(X) / len(indices)
    return loss, acc


def train_models(
    model: torch.nn.Module, optimizer: str, lrs: list[float], train_ds, test_loader: Loader, batch_size: int, steps: int, device: str,
) -> list[dict]:
    """Train one model per lr (vmapped if several) and evaluate each and its power-function EMA on the test set"""
    batched = len(lrs) > 1
    magic = Magic(objective=Objective(model=model, loss_fn=LOSSES["ce"]), loader=Loader(train_ds, batch_size, device=device), compile=device.startswith("cuda"))
    params = {k: p.detach().clone() for k, p in model.named_parameters()}
    weights = None
    if not batched:
        state = (AdamState if optimizer == "adam" else SGDState)(params=params, lr=LRSchedule(lrs[0]))
    elif optimizer == "adam":  # the lr of each model in the per-tensor lr multipliers
        state = AdamState(params=params, lr=LRSchedule(1.0))
        state.lr_mults = {k: mult * torch.tensor(lrs, device=device).view(-1, *[1] * params[k].ndim) for k, mult in state.lr_mults.items()}
        state = state.apply(lambda p: p.expand(len(lrs), *p.shape).clone())
    else:  # sgd with lr 1 on lr * loss
        state = SGDState(params=params, lr=LRSchedule(1.0))
        weights = torch.tensor(lrs)[:, None].expand(len(lrs), len(train_ds)).contiguous()

    test_ds = test_loader.dataset
    all_test, some_test = torch.arange(len(test_ds)), test_ds.subset(min(2000, len(test_ds)))
    ema, last_params, curve = None, state.params, {}
    sync(device)
    start = time.perf_counter()
    for t, state_t, grads in magic.train.trace(state, n_steps=steps, weights=weights, pbar=False):
        last_params = {k: p.detach() for k, p in state_t.params.items()}
        beta = (1 - 1 / (t + 1)) ** (EMA_GAMMA + 1)
        ema = last_params if ema is None else {k: e + (1 - beta) * (p - e) for k, (e, p) in dict_zip(ema, last_params).items()}
        if t and t % max(1, steps // 10) == 0 and grads is not None:
            curve[t] = evaluate(magic, ema, test_loader, some_test, batched)[1].reshape(-1).tolist()
    sync(device)
    seconds = time.perf_counter() - start
    finals = {name: evaluate(magic, p, test_loader, all_test, batched) for name, p in (("test", last_params), ("ema_test", ema))}
    rows = []
    for i, lr in enumerate(lrs):
        row = {"lr": lr, "train_s": round(seconds, 1), "ema_test_acc_curve": {t: round(accs[i], 4) for t, accs in curve.items()}}
        for name, (loss, acc) in finals.items():
            loss_i, acc_i = loss.reshape(-1)[i].item(), acc.reshape(-1)[i].item()
            row |= {f"{name}_loss": round(loss_i, 4) if np.isfinite(loss_i) else None, f"{name}_acc": round(acc_i, 4) if np.isfinite(loss_i) else None}
        rows.append(row)
    return rows


def best_lr(rows: list[dict]) -> float:
    """The lr with the lowest EMA test loss"""
    return min((r for r in rows if r["ema_test_loss"] is not None), key=lambda r: r["ema_test_loss"])["lr"]


@click.group(help=__doc__)
def cli():
    pass


@cli.command()
@click.argument("out_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--model", type=click.Choice(list(MODELS)), required=True)
@click.option("--optimizer", type=click.Choice(["sgd", "adam"]), required=True)
@click.option("--steps", type=int, default=10000, show_default=True)
@click.option("--batch-sizes", type=str, default=",".join(map(str, BATCH_SIZES)), show_default=True)
@click.option("--device", type=str, default="cuda", show_default=True)
def run(out_dir, model, optimizer, steps, batch_sizes, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model}_{optimizer}.jsonl"
    train_ds, test_ds = DATASETS["cifar10"](train=True).to(device), DATASETS["cifar10"](train=False).to(device)
    test_loader = Loader(test_ds, 1000, device=device)
    width, depth = MODELS[model]
    for batch_size in sorted((int(b) for b in batch_sizes.split(",")), reverse=True):  # the slowest tuning first
        for role, model_width in (("tune", width // 2), ("real", width)):
            done = [r for r in read_jsonl(path) if r["batch_size"] == batch_size and r["steps"] == steps]
            if any(r["role"] == role for r in done):
                continue
            lrs = lr_grid(model, optimizer, batch_size) if role == "tune" else [best_lr([r for r in done if r["role"] == "tune"])]
            net = build_model(model, model_width, depth, train_ds.in_shape, train_ds.out_dim, device)
            common = {"model": model, "optimizer": optimizer, "role": role, "width": model_width, "depth": depth,
                      "n_params": sum(p.numel() for p in net.parameters()), "batch_size": batch_size, "steps": steps}
            for row in train_models(net, optimizer, lrs, train_ds, test_loader, batch_size, steps, device):
                append_jsonl(path, common | row)
            torch.compiler.reset()


@cli.command()
@click.argument("out_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--show/--no-show", default=False)
def plot(out_dir, show):
    rows = [r for path in sorted(out_dir.glob("*.jsonl")) for r in read_jsonl(path)]
    runs = sorted({(r["model"], r["optimizer"]) for r in rows})
    batch_sizes = sorted({r["batch_size"] for r in rows})
    steps = max(r["steps"] for r in rows)
    batch_colors = dict(zip(batch_sizes, plt.cm.viridis(np.linspace(0.9, 0, len(batch_sizes)))))

    fig, axes = plt.subplots(2, len(runs), figsize=(4.6 * len(runs), 7), squeeze=False, layout="constrained")
    for col, (model, optimizer) in enumerate(runs):
        tuned = [r for r in rows if (r["model"], r["optimizer"], r["role"]) == (model, optimizer, "tune")]
        for row, name in enumerate(("ema_test_loss", "ema_test_acc")):
            ax = axes[row, col]
            for batch_size in batch_sizes:
                points = sorted((r["lr"], r[name]) for r in tuned if r["batch_size"] == batch_size and r[name] is not None)
                if not points:
                    continue
                ax.plot(*zip(*points), "o-", ms=3, lw=1.2, color=batch_colors[batch_size], label=f"B = {batch_size}")
                chosen = best_lr([r for r in tuned if r["batch_size"] == batch_size])
                ax.plot(chosen, dict(points)[chosen], "D", ms=8, mfc="none", mec="k", mew=1)
            width = tuned[0]["width"] if tuned else ""
            ax.set(xscale="log", xlabel="lr", ylabel=name.replace("_", " ").replace("ema", "EMA"), title=f"{model} {width}x{MODELS[model][1]} (half width), {optimizer}")
            ax.title.set_fontsize(9)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, frameon=False)
    fig_title(fig, f"lr tuning on the half-width models, {steps} steps (diamond: the lowest EMA test loss, used for the real model; missing points: diverged)")
    fig.savefig(out_dir / "lr_tuning.png", dpi=150)

    colors = {"sgd": "tab:blue", "adam": "tab:orange"}
    markers = {"cnn": "o", "vit": "s"}
    panels = [("lr", "best lr (from the half-width model)", "log"), ("ema_test_loss", "EMA test loss", "linear"), ("ema_test_acc", "EMA test acc", "linear"),
              ("test_acc", "test acc (last params)", "linear")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.9), squeeze=False, layout="constrained")
    summary = ["| model | optimizer | B | best lr | EMA test loss | EMA test acc | test acc | half-width EMA test acc | train s |", "|---|---|---|---|---|---|---|---|---|"]
    for model, optimizer in runs:
        real = sorted((r for r in rows if (r["model"], r["optimizer"], r["role"]) == (model, optimizer, "real")), key=lambda r: r["batch_size"])
        name = f"{model} {MODELS[model][0]}x{MODELS[model][1]}, {optimizer}"
        for ax, (key, label, yscale) in zip(axes[0], panels):
            points = [(r["batch_size"], r[key]) for r in real if r[key] is not None]
            if points:
                ax.plot(*zip(*points), marker=markers[model], ms=5, lw=1.3, color=colors[optimizer], ls="-" if model == "cnn" else "--", label=name)
            ax.set(xscale="log", yscale=yscale, xlabel="batch size", ylabel=label)
            ax.set_xticks(batch_sizes, [str(b) for b in batch_sizes])
            ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
            ax.grid(alpha=0.2)
        for r in real:
            small = next((s for s in rows if (s["model"], s["optimizer"], s["role"], s["batch_size"], s["lr"]) == (model, optimizer, "tune", r["batch_size"], r["lr"])), {})
            cell = lambda v: "diverged" if v is None else f"{v:.3f}"
            summary.append(f"| {model} {r['width']}x{r['depth']} | {optimizer} | {r['batch_size']} | {r['lr']:g} | {cell(r['ema_test_loss'])} | {cell(r['ema_test_acc'])} | "
                           f"{cell(r['test_acc'])} | {cell(small.get('ema_test_acc'))} | {r['train_s']:.0f} |")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig_title(fig, f"The real models after {steps} steps at the lr tuned per batch size (EMA: gamma {EMA_GAMMA:g})")
    fig.savefig(out_dir / "batch_size.png", dpi=150)
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n")
    click.echo("\n".join(summary))
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    cli()
