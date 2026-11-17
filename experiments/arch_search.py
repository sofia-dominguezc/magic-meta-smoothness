"""
Which model size, and how large a batch before a step gets slower? On CIFAR10, over CNN and ViT sizes (width x depth):
  python -m experiments.arch_search time OUT [--device cuda]    # step time vs batch size, and vs vmapped models
  python -m experiments.arch_search train OUT [--device cuda]   # 1000 steps at B = 128 per lr: test acc of the EMA model
  python -m experiments.arch_search plot OUT                    # the plots and OUT/summary.md

`time` appends to OUT/timing.jsonl and `train` to OUT/train.jsonl (both skip what is already there). Plots:
- accuracy.png: EMA test acc at the best lr vs the number of params, and vs the step time at B = 128
- lr.png: EMA test acc vs lr, per model
- batch_<optimizer>_<mode>.png: ms / step and examples / s vs the batch size, per model (mode: eager, or compiled forward)
- memory.png: peak GPU memory of a step vs the batch size (with the dataset and the state: ~0.6 GB), and per extra example
- vmap_<mode>.png: examples / s of R vmapped models at B = 128, and relative to one model at B = 128 R
"""

import json
import time
from functools import partial
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import DATASETS, Loader
from src.logging import fig_title
from src.magic import Magic
from src.models import MODELS
from src.optimizers import AdamState, LRSchedule, OptimState, SGDState
from src.utils import LOSSES, dict_zip

# (model, width, depth): 0.3M to 5M params (depth: blocks per stage of the cnn)
ARCHS = [
    ("cnn", 32, 1), ("cnn", 48, 1), ("cnn", 32, 2), ("cnn", 64, 1), ("cnn", 48, 2), ("cnn", 96, 1), ("cnn", 64, 2), ("cnn", 64, 3), ("cnn", 128, 1),
    ("vit", 96, 3), ("vit", 96, 6), ("vit", 128, 4), ("vit", 192, 3), ("vit", 128, 8), ("vit", 192, 6), ("vit", 256, 4), ("vit", 256, 6),
]
LRS = {"sgd": [0.1, 0.05, 0.2, 0.02, 0.5], "adam": [0.05, 0.02, 0.1, 0.01, 0.2]}  # tried in this order
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
N_MODELS = [1, 2, 4, 8, 16, 32]
VMAP_BATCH = 128
EMA_GAMMA = 20.0
FLAT_TOLERANCE = 1.3  # a step is "not slower" up to this factor of the fastest one
SATURATED = 0.9  # fraction of the peak examples / s


def arch_name(model: str, width: int, depth: int) -> str:
    return f"{model} {width}x{depth}"


def build_model(model: str, width: int, depth: int, in_shape: tuple[int, ...], out_dim: int, device: str) -> torch.nn.Module:
    torch.manual_seed(0)
    return MODELS[model](in_shape, out_dim, width=width, **{"blocks" if model == "cnn" else "depth": depth}).to(device)


def init_state(model: torch.nn.Module, optimizer: str, lr: float) -> OptimState:
    params = {k: p.detach().clone() for k, p in model.named_parameters()}
    return (AdamState if optimizer == "adam" else SGDState)(params=params, lr=LRSchedule(lr))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def append_jsonl(path: Path, row: dict):
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    click.echo(json.dumps(row))


def sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def time_steps(forward, state: OptimState, loader: Loader, device: str, min_seconds: float = 0.5, max_steps: int = 30) -> dict:
    """Training steps as in `Magic.train`: the first step's seconds (with any compilation) and the mean ms of the later ones"""
    def run(steps: range, state: OptimState) -> OptimState:
        for t in steps:
            X, Y, _ = loader.get_batch(t)
            for p in state.params.values():
                p.requires_grad_()
            state = state.step(t, partial(forward, X=X, Y=Y, weights=None))[0].apply(lambda p: p.detach())
        sync(device)
        return state

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    state = run(range(1), state)
    first_step = time.perf_counter() - start
    state = run(range(1, 4), state)
    n_steps, start = 0, time.perf_counter()
    while n_steps < 10 or (n_steps < max_steps and time.perf_counter() - start < min_seconds):
        state = run(range(4 + n_steps, 9 + n_steps), state)
        n_steps += 5
    ms = 1e3 * (time.perf_counter() - start) / n_steps
    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else 0.0
    return {"ms_per_step": round(ms, 3), "first_step_s": round(first_step, 2), "peak_gb": round(peak_gb, 2)}


@click.group(help=__doc__)
def cli():
    pass


@cli.command(name="time")
@click.argument("out_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--compile-budget", type=float, default=20, show_default=True, help="stop compiling after this many minutes of compilation")
@click.option("--archs", type=str, default=None, help='only these, e.g. "cnn 32x1,vit 96x3"')
@click.option("--max-batch", type=int, default=BATCH_SIZES[-1], show_default=True)
@click.option("--compiled-vmap/--no-compiled-vmap", default=True, help="also time the vmapped models with the compiled forward (a compilation per number of models)")
def time_command(out_dir, device, compile_budget, archs, max_batch, compiled_vmap):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "timing.jsonl"
    done = {(r["arch"], r["optimizer"], r["mode"], r["batch_size"], r["n_models"]) for r in read_jsonl(path)}
    train_ds = DATASETS["cifar10"](train=True).to(device)
    compile_seconds = 0.0
    for model_name, width, depth in ARCHS:
        arch = arch_name(model_name, width, depth)
        if archs is not None and arch not in archs.split(","):
            continue
        model = build_model(model_name, width, depth, train_ds.in_shape, train_ds.out_dim, device)
        n_params = sum(p.numel() for p in model.parameters())
        magic = Magic(model=model, loader=Loader(train_ds, VMAP_BATCH, device=device), loss_fn=LOSSES["ce"])
        vmapped = torch.func.vmap(magic.forward, in_dims=(0, None, None, None))
        vmapped_forward = lambda params, X, Y, weights=None: vmapped(params, X, Y, weights)
        forwards = {("eager", False): magic.forward, ("eager", True): vmapped_forward}
        if compile_seconds < 60 * compile_budget:
            forwards |= {("compiled", False): torch.compile(magic.forward), ("compiled", True): torch.compile(vmapped_forward)}
        # (optimizer, mode, batch size, vmapped models): adam's compiled forward is sgd's
        jobs = [(opt, mode, B, 1) for opt, mode in (("sgd", "eager"), ("sgd", "compiled"), ("adam", "eager")) for B in BATCH_SIZES if B <= max_batch]
        jobs += [("sgd", mode, VMAP_BATCH, R) for mode in ("eager", "compiled")[: 2 if compiled_vmap else 1] for R in N_MODELS[1:] if VMAP_BATCH * R <= max_batch]
        failed: set[tuple[str, str, bool]] = set()  # out of memory: skip the larger ones
        for opt, mode, batch_size, n_models in jobs:
            key = (opt, mode, n_models > 1)
            if (arch, opt, mode, batch_size, n_models) in done or key in failed or (mode, n_models > 1) not in forwards:
                continue
            state = init_state(model, opt, 1e-3)
            if n_models > 1:
                state = state.apply(lambda p: p.expand(n_models, *p.shape).clone())
            row = {"arch": arch, "model": model_name, "width": width, "depth": depth, "n_params": n_params, "optimizer": opt, "mode": mode,
                   "batch_size": batch_size, "n_models": n_models}
            try:
                row |= time_steps(forwards[mode, n_models > 1], state, Loader(train_ds, batch_size, device=device), device)
                row["examples_per_s"] = round(1e3 * batch_size * n_models / row["ms_per_step"], 1)
                if mode == "compiled":
                    compile_seconds += row["first_step_s"]
            except Exception as e:  # out of memory, or a failed compilation
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                failed.add(key)
                del state
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            append_jsonl(path, row)
        torch.compiler.reset()


def train_run(magic: Magic, state: OptimState, test_loader: Loader, n_steps: int, device: str) -> dict:
    """Train and evaluate the last params and their power-function EMA (as in `main.train_logged`) on the test set"""
    test_ds = test_loader.dataset
    all_test, some_test = torch.arange(len(test_ds)), test_ds.subset(min(2000, len(test_ds)))
    ema, params, curve = None, state.params, {}
    sync(device)
    start = time.perf_counter()
    for t, state_t, grads in magic.train.trace(state, n_steps=n_steps, pbar=False):
        params = {k: p.detach() for k, p in state_t.params.items()}
        beta = (1 - 1 / (t + 1)) ** (EMA_GAMMA + 1)
        ema = params if ema is None else {k: e + (1 - beta) * (p - e) for k, (e, p) in dict_zip(ema, params).items()}
        if t % 50 == 0 and not all(torch.isfinite(p).all() for p in params.values()):
            break
        if t and t % (n_steps // 4) == 0 and grads is not None:
            curve[t] = round(magic.eval_loss(ema, test_loader, some_test)[1]["acc"].item(), 4)
    sync(device)
    seconds = time.perf_counter() - start
    row = {"train_s": round(seconds, 2), "ema_test_acc_curve": curve}
    for name, p in (("test", params), ("ema_test", ema)):
        loss, mean_metrics = magic.eval_loss(p, test_loader, all_test)
        finite = bool(torch.isfinite(loss))
        row |= {f"{name}_loss": round(loss.item(), 4) if finite else None, f"{name}_acc": round(mean_metrics["acc"].item(), 4) if finite else None}
    return row


@cli.command()
@click.argument("out_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--steps", type=int, default=1000, show_default=True)
@click.option("--batch-size", type=int, default=128, show_default=True)
@click.option("--budget", type=float, default=60, show_default=True, help="stop starting runs after this many minutes")
@click.option("--archs", type=str, default=None, help='only these, e.g. "cnn 32x1,vit 96x3"')
@click.option("--models", type=str, default="cnn,vit", show_default=True)
@click.option("--sgd-lrs", type=str, default=",".join(map(str, LRS["sgd"])), show_default=True, help="tried in this order")
@click.option("--adam-lrs", type=str, default=",".join(map(str, LRS["adam"])), show_default=True)
def train(out_dir, device, steps, batch_size, budget, archs, models, sgd_lrs, adam_lrs):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "train.jsonl"
    done = {(r["arch"], r["optimizer"], r["lr"]) for r in read_jsonl(path)}
    train_ds, test_ds = DATASETS["cifar10"](train=True).to(device), DATASETS["cifar10"](train=False).to(device)
    loader, test_loader = Loader(train_ds, batch_size, device=device), Loader(test_ds, 1000, device=device)
    lrs = {opt: [float(lr) for lr in text.split(",") if lr] for opt, text in (("sgd", sgd_lrs), ("adam", adam_lrs))}
    start = time.perf_counter()
    for lr_index in range(max(map(len, lrs.values()))):  # every model at the first lrs first
        for model_name, width, depth in ARCHS:
            arch = arch_name(model_name, width, depth)
            if (archs is not None and arch not in archs.split(",")) or model_name not in models.split(","):
                continue
            model = build_model(model_name, width, depth, train_ds.in_shape, train_ds.out_dim, device)
            magic = Magic(model=model, loader=loader, loss_fn=LOSSES["ce"])
            for opt in ("sgd", "adam"):
                if lr_index >= len(lrs[opt]):
                    continue
                lr = lrs[opt][lr_index]
                if (arch, opt, lr) in done:
                    continue
                if time.perf_counter() - start > 60 * budget:
                    click.echo(f"out of budget at lr index {lr_index}, {arch}")
                    return
                row = {"arch": arch, "model": model_name, "width": width, "depth": depth, "n_params": sum(p.numel() for p in model.parameters()),
                       "optimizer": opt, "lr": lr, "steps": steps, "batch_size": batch_size}
                append_jsonl(path, row | train_run(magic, init_state(model, opt, lr), test_loader, steps, device))


def best_runs(train_rows: list[dict]) -> dict[tuple[str, str], dict]:
    """(arch, optimizer) -> its run with the lowest EMA test loss"""
    best: dict[tuple[str, str], dict] = {}
    for r in train_rows:
        key = (r["arch"], r["optimizer"])
        if r["ema_test_loss"] is not None and (key not in best or r["ema_test_loss"] < best[key]["ema_test_loss"]):
            best[key] = r
    return best


def batch_thresholds(rows: list[dict]) -> dict:
    """Of one model's timings by batch size: the largest batch whose step is within FLAT_TOLERANCE of the fastest step, and
    the smallest one at SATURATED of the peak examples / s"""
    rows = sorted(rows, key=lambda r: r["batch_size"])
    fastest, peak = min(r["ms_per_step"] for r in rows), max(r["examples_per_s"] for r in rows)
    flat = max(r["batch_size"] for r in rows if r["ms_per_step"] <= FLAT_TOLERANCE * fastest)
    saturated = min(r["batch_size"] for r in rows if r["examples_per_s"] >= SATURATED * peak)
    return {"flat_batch": flat, "saturated_batch": saturated, "peak_examples_per_s": peak, "fastest_ms": fastest}


def arch_colors(rows: list[dict]) -> dict[str, tuple]:
    """A color per model of `rows`, light to dark by its number of params"""
    sizes = sorted({(r["n_params"], r["arch"]) for r in rows})
    return {arch: color for (_, arch), color in zip(sizes, plt.cm.viridis(np.linspace(0.9, 0, len(sizes))))}


def label_line_ends(ax, ends: list[tuple[float, float, str]]):
    """Label each line at its last point (x, y, text), small and gray"""
    for x, y, text in ends:
        ax.annotate(text, (x, y), xytext=(3, 0), textcoords="offset points", fontsize=6, color="0.3", va="center")


def plot_accuracy(best: dict[tuple[str, str], dict], step_ms: dict[str, float], path: Path):
    colors = {"sgd": "tab:blue", "adam": "tab:orange"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False, layout="constrained")
    for col, model in enumerate(("cnn", "vit")):
        for row, (x_name, x_label) in enumerate((("n_params", "parameters"), ("ms", f"ms / step at B = {VMAP_BATCH} (sgd, compiled forward)"))):
            ax = axes[row, col]
            for opt, color in colors.items():
                runs = sorted((r for (_, o), r in best.items() if o == opt and r["model"] == model), key=lambda r: r["n_params"])
                xs = [r["n_params"] if x_name == "n_params" else step_ms.get(r["arch"], float("nan")) for r in runs]
                ax.plot(xs, [r["ema_test_acc"] for r in runs], "o", ms=5, color=color, label=opt)
                for x, r in zip(xs, runs):
                    ax.annotate(f"{r['width']}x{r['depth']}", (x, r["ema_test_acc"]), xytext=(4, 3), textcoords="offset points", fontsize=6, color="0.3")
            ax.set(xscale="log", xlabel=x_label, ylabel="EMA test acc", title=model)
            ax.grid(alpha=0.2)
            ax.legend(title="optimizer", fontsize=8, frameon=False)
    fig_title(fig, f"CIFAR10, 1000 steps at B = 128, best lr per point: test acc of the EMA model (gamma {EMA_GAMMA:g}); labels: width x depth")
    fig.savefig(path, dpi=150)


def plot_lrs(train_rows: list[dict], path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False, layout="constrained")
    for row, model in enumerate(("cnn", "vit")):
        colors = arch_colors([r for r in train_rows if r["model"] == model])
        for col, opt in enumerate(("sgd", "adam")):
            ax = axes[row, col]
            for arch, color in colors.items():
                runs = sorted((r for r in train_rows if r["arch"] == arch and r["optimizer"] == opt and r["ema_test_acc"] is not None), key=lambda r: r["lr"])
                ax.plot([r["lr"] for r in runs], [r["ema_test_acc"] for r in runs], "o-", ms=3, lw=1, color=color, label=arch)
            ax.set(xscale="log", xlabel="lr", ylabel="EMA test acc", title=f"{model}, {opt}")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=6, frameon=False)
    fig_title(fig, "EMA test acc vs lr (missing points: diverged)")
    fig.savefig(path, dpi=150)


def plot_batch(rows: list[dict], title: str, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False, layout="constrained")
    for row, model in enumerate(("cnn", "vit")):
        model_rows = [r for r in rows if r["model"] == model]
        colors = arch_colors(model_rows)
        for col, (name, label) in enumerate((("ms_per_step", "ms / step"), ("examples_per_s", "examples / s"))):
            ax, ends = axes[row, col], []
            for arch, color in colors.items():
                runs = sorted((r for r in model_rows if r["arch"] == arch), key=lambda r: r["batch_size"])
                ax.plot([r["batch_size"] for r in runs], [r[name] for r in runs], "o-", ms=3, lw=1.2, color=color, label=arch)
                thresholds = batch_thresholds(runs)
                mark = thresholds["flat_batch"] if name == "ms_per_step" else thresholds["saturated_batch"]
                ax.plot(mark, next(r[name] for r in runs if r["batch_size"] == mark), "D", ms=7, mfc="none", mec="k", mew=1)
                ends.append((runs[-1]["batch_size"], runs[-1][name], arch.split()[1]))
            label_line_ends(ax, ends)
            marked = f"largest batch within {FLAT_TOLERANCE:g}x of the fastest step" if name == "ms_per_step" else f"smallest batch at {SATURATED:.0%} of the peak"
            ax.set(xscale="log", yscale="log", xlabel="batch size", ylabel=label, title=f"{model}: {label} (diamond: {marked})")
            ax.title.set_fontsize(9)
            ax.set_xticks(BATCH_SIZES, [str(b) for b in BATCH_SIZES])
            ax.tick_params(axis="x", which="minor", bottom=False)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=6, frameon=False, title="width x depth", title_fontsize=7)
    fig_title(fig, title)
    fig.savefig(path, dpi=150)


def memory_per_example(rows: list[dict]) -> list[tuple[int, float]]:
    """Of one model's timings by batch size: (batch size, MB of peak memory per example added since the previous batch size)"""
    rows = sorted(rows, key=lambda r: r["batch_size"])
    return [(b["batch_size"], 1024 * (b["peak_gb"] - a["peak_gb"]) / (b["batch_size"] - a["batch_size"])) for a, b in zip(rows, rows[1:])]


def plot_memory(rows: list[dict], title: str, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False, layout="constrained")
    for row, model in enumerate(("cnn", "vit")):
        model_rows = [r for r in rows if r["model"] == model]
        colors = arch_colors(model_rows)
        for col, label in enumerate(("peak GPU memory (GB)", "MB per extra example")):
            ax, ends = axes[row, col], []
            for arch, color in colors.items():
                runs = sorted((r for r in model_rows if r["arch"] == arch), key=lambda r: r["batch_size"])
                points = [(r["batch_size"], r["peak_gb"]) for r in runs] if col == 0 else memory_per_example(runs)
                ax.plot(*zip(*points), "o-", ms=3, lw=1.2, color=color, label=arch)
                ends.append((*points[-1], arch.split()[1]))
            label_line_ends(ax, ends)
            ax.set(xscale="log", yscale="log", xlabel="batch size", ylabel=label, title=f"{model}: {label}")
            ax.title.set_fontsize(9)
            ax.set_xticks(BATCH_SIZES, [str(b) for b in BATCH_SIZES])
            ax.tick_params(axis="x", which="minor", bottom=False)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=6, frameon=False, title="width x depth", title_fontsize=7)
    fig_title(fig, title)
    fig.savefig(path, dpi=150)


def plot_vmap(rows: list[dict], title: str, path: Path):
    """`rows`: one optimizer and mode, single models (by batch size) and vmapped ones (by number of models)"""
    single = {(r["arch"], r["batch_size"]): r["examples_per_s"] for r in rows if r["n_models"] == 1}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False, layout="constrained")
    for row, model in enumerate(("cnn", "vit")):
        model_rows = [r for r in rows if r["model"] == model]
        colors = arch_colors(model_rows)
        for col in range(2):
            ax, ends = axes[row, col], []
            for arch, color in colors.items():
                runs = sorted((r for r in model_rows if r["arch"] == arch and (r["n_models"] > 1 or r["batch_size"] == VMAP_BATCH)), key=lambda r: r["n_models"])
                runs = [r for r in runs if (arch, VMAP_BATCH * r["n_models"]) in single]
                if not runs:
                    continue
                n_models = [r["n_models"] for r in runs]
                one_model = [single[arch, VMAP_BATCH * r["n_models"]] for r in runs]
                values = [r["examples_per_s"] for r in runs] if col == 0 else [r["examples_per_s"] / s for r, s in zip(runs, one_model)]
                ax.plot(n_models, values, "o-", ms=3, lw=1.2, color=color, label=arch)
                if col == 0:
                    ax.plot(n_models, one_model, "--", lw=0.9, color=color)
                ends.append((n_models[-1], values[-1], arch.split()[1]))
            label_line_ends(ax, ends)
            if col == 1:
                ax.axhline(1, color="gray", lw=0.8, ls=":")
            ax.set(xscale="log", yscale="log", xlabel=f"R = vmapped models, each at B = {VMAP_BATCH}",
                   ylabel="examples / s" if col == 0 else "ratio",
                   title=f"{model}: examples / s of R vmapped models (dashed: one model at B = {VMAP_BATCH} R)" if col == 0
                   else f"{model}: R vmapped models / one model at B = {VMAP_BATCH} R")
            ax.title.set_fontsize(9)
            ax.set_xticks(N_MODELS, [str(n) for n in N_MODELS])
            ax.tick_params(axis="x", which="minor", bottom=False)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=6, frameon=False, title="width x depth", title_fontsize=7)
    fig_title(fig, title)
    fig.savefig(path, dpi=150)


@cli.command()
@click.argument("out_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--show/--no-show", default=False)
def plot(out_dir, show):
    timing = [r for r in read_jsonl(out_dir / "timing.jsonl") if "error" not in r]
    train_rows = read_jsonl(out_dir / "train.jsonl")
    best = best_runs(train_rows)

    def timings(opt: str, mode: str, vmapped: bool | None = False) -> list[dict]:
        return [r for r in timing if r["optimizer"] == opt and r["mode"] == mode and (vmapped is None or (r["n_models"] > 1) == vmapped)]

    step_mode = "compiled" if timings("sgd", "compiled") else "eager"
    step_ms = {r["arch"]: r["ms_per_step"] for r in timings("sgd", step_mode) if r["batch_size"] == VMAP_BATCH}
    if best:
        plot_accuracy(best, step_ms, out_dir / "accuracy.png")
        plot_lrs(train_rows, out_dir / "lr.png")
    for opt, mode in (("sgd", "eager"), ("sgd", "compiled"), ("adam", "eager")):
        if timings(opt, mode):
            forward = "compiled forward" if mode == "compiled" else "eager"
            plot_batch(timings(opt, mode), f"One training step vs the batch size: {opt}, {forward}", out_dir / f"batch_{opt}_{mode}.png")
    if timings("sgd", "eager"):
        plot_memory(timings("sgd", "eager"), "Peak GPU memory of a training step (sgd, eager; the dataset and the state included)", out_dir / "memory.png")
    for mode in ("eager", "compiled"):
        if timings("sgd", mode, vmapped=True):
            forward = "compiled forward" if mode == "compiled" else "eager"
            plot_vmap(timings("sgd", mode, vmapped=None), f"Vmapped models vs a larger batch: sgd, {forward}", out_dir / f"vmap_{mode}.png")

    archs = {r["arch"]: r for r in timing + train_rows}
    lines = ["| model | params | sgd acc (lr) | adam acc (lr) | ms/step B=128 eager | compiled | flat batch eager | compiled | 90% peak batch eager | compiled | peak ex/s compiled | vmap R=8 / B=1024 eager | compiled | peak GB B=128 | B=1024 | MB / example |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arch, r in sorted(archs.items(), key=lambda item: (item[1]["model"], item[1]["n_params"])):
        cells = [arch, f"{r['n_params'] / 1e6:.2f}M"]
        for opt in ("sgd", "adam"):
            run = best.get((arch, opt))
            cells.append("" if run is None else f"{run['ema_test_acc']:.3f} ({run['lr']:g})")
        by_mode = {mode: [t for t in timings("sgd", mode) if t["arch"] == arch] for mode in ("eager", "compiled")}
        thresholds = {mode: batch_thresholds(rows) if rows else None for mode, rows in by_mode.items()}
        cells += [next((f"{t['ms_per_step']:.1f}" for t in by_mode[mode] if t["batch_size"] == VMAP_BATCH), "") for mode in ("eager", "compiled")]
        for name in ("flat_batch", "saturated_batch"):
            cells += ["" if thresholds[mode] is None else str(thresholds[mode][name]) for mode in ("eager", "compiled")]
        cells.append("" if thresholds["compiled"] is None else f"{thresholds['compiled']['peak_examples_per_s']:.0f}")
        for mode in ("eager", "compiled"):
            vmapped = next((t["examples_per_s"] for t in timings("sgd", mode, vmapped=True) if t["arch"] == arch and t["n_models"] == 8), None)
            one_model = next((t["examples_per_s"] for t in by_mode[mode] if t["batch_size"] == 8 * VMAP_BATCH), None)
            cells.append("" if vmapped is None or one_model is None else f"{vmapped / one_model:.2f}")
        cells += [next((f"{t['peak_gb']:.2f}" for t in by_mode["eager"] if t["batch_size"] == B), "") for B in (128, 1024)]
        cells.append(f"{memory_per_example(by_mode['eager'])[-1][1]:.1f}" if len(by_mode["eager"]) > 1 else "")
        lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    click.echo("\n".join(lines))
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    cli()
