"""

How different are the models trained on all the data by the plain and the vmapped training passes, and does it grow
with the steps?
  python -m experiments.vmap_error DIR [--device cpu] [--no-show]

Trains the experiment's full model (config in DIR: same init and batches, all the weights 1; the model forward in
bfloat16 if the config's dtype is bf16, as in retrain) with fp64 and with fp32 states, each
- "plain": `Magic.train` without weights (torch.func.grad, no vmap), as run-magic's reference f(s_T)
- "vmap":  `Magic.train` vmapped with 2 copies (row 0 is taken; the rows are identical with deterministic kernels),
           as retrain does
Saves DIR/vmap_error.png: per dtype (columns, same axes), the operator norms (`logging.op_norm`, like the other norm
plots: raw parameters are O(1) in it) of the differences vmap - plain of each parameter along training, and the
absolute difference of their test losses (on the same 1000 test examples as train's test_loss), in log scale, plus
a third column with the same for fp32 plain - fp64 plain (the fp32 rounding of the states and kernels);
DIR/vmap_error_linear.png: the same in linear scale over the second half of the steps; and DIR/vmap_error.json: these
values, the test losses and the peak GPU memory of each run (--replot redoes the plots from it).
The differences are floating-point rounding (batched and unbatched matmuls use different kernels) amplified by the
training dynamics: they grow with the steps, a lot when training is chaotic (edge of stability).
"""

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

from main import DTYPES, Experiment
from src.data import Subset
from src.logging import op_norm
from src.magic import eval_loss


def vmap_error(exp_dir: Path, device: str = "cpu", show: bool = True) -> dict:
    exp = Experiment(exp_dir, device)
    cfg = exp.cfg
    ckpt_every = max(1, cfg.n_steps // 50)
    steps = [*range(0, cfg.n_steps, ckpt_every), cfg.n_steps]
    test_loader = exp.eval_loader(Subset(n=1000))
    torch.set_default_dtype(torch.float64)
    model, init_state = exp.build()  # built once: the fp32 and fp64 samplers draw different inits from the same seed
    results: dict = {"steps": steps}
    runs: dict = {}  # dtype -> "plain" / "vmap" -> states [len(steps)]
    for dtype in ["fp64", "fp32"]:
        torch.set_default_dtype(DTYPES[dtype])
        magic = exp.magic(model)
        runs[dtype], res = {}, {"test_loss": {}, "peak_gpu_mb": {}}
        for name in ["plain", "vmap"]:
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            final, ckpts = magic.train(
                init_state.apply(lambda t: t.to(DTYPES[dtype])), n_steps=cfg.n_steps, ckpt_every=ckpt_every,
                weights=torch.ones(2, len(exp.train_ds)) if name == "vmap" else None,
            )
            states = [*ckpts, final]
            runs[dtype][name] = [s.apply(lambda t: t[0]) for s in states] if name == "vmap" else states
            res["peak_gpu_mb"][name] = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else float("nan")
            res["test_loss"][name] = [
                eval_loss(model, exp.loss_fn, {k: p.to(device) for k, p in s.params.items()}, test_loader, bf16=exp.bf16)[0].item()
                for s in runs[dtype][name]
            ]
            click.echo(f"{dtype} {name:5s} test loss {res['test_loss'][name][-1]:.6f}  peak GPU {res['peak_gpu_mb'][name]:.0f} MB")
        res["norms"] = norms_of(runs[dtype]["vmap"], runs[dtype]["plain"])
        results[dtype] = res
        (exp_dir / "vmap_error.json").write_text(json.dumps(results, indent=2))  # the fp64 half is the slow one
    results["fp32 plain - fp64 plain"] = {
        "norms": norms_of(runs["fp32"]["plain"], runs["fp64"]["plain"]),
        "test_loss_diff": (np.array(results["fp32"]["test_loss"]["plain"]) - np.array(results["fp64"]["test_loss"]["plain"])).tolist(),
    }
    (exp_dir / "vmap_error.json").write_text(json.dumps(results, indent=2))
    plot(exp_dir, results, exp.name, show)
    return results


def norms_of(a: list, b: list) -> dict[str, list[float]]:
    """{param: op norms of a - b per step} for the states `a` and `b` [n_steps]."""
    norms, us = {k: [] for k in a[0].params}, {}
    for s, t in zip(a, b):
        for k in norms:
            norm, us[k] = op_norm(s.params[k].double() - t.params[k].double(), us.get(k), n_iter=3 if k in us else 10)
            norms[k].append(norm.item())
    return norms


def plot(exp_dir: Path, results: dict, title: str, show: bool = True):
    """DIR/vmap_error.png (log scale, all the steps) and DIR/vmap_error_linear.png (linear, the second half)."""
    steps = np.array(results["steps"])
    layers = list(dict.fromkeys(k.rpartition(".")[0] or k for k in results["fp64"]["norms"]))
    colors = dict(zip(layers, plt.cm.viridis(np.linspace(0, 0.9, len(layers)))))
    columns = [  # (title, {param: norms}, loss differences)
        (f"{dtype}: vmap - plain", results[dtype]["norms"],
         np.array(results[dtype]["test_loss"]["vmap"]) - np.array(results[dtype]["test_loss"]["plain"]))
        for dtype in ["fp64", "fp32"]
    ]
    if (cross := results.get("fp32 plain - fp64 plain")) is not None:
        columns.append(("fp32 plain - fp64 plain", cross["norms"], np.array(cross["test_loss_diff"])))
    for name, log, sel in [("vmap_error", True, steps >= 0), ("vmap_error_linear", False, steps >= steps[-1] // 2)]:
        fig, axes = plt.subplots(2, len(columns), figsize=(6 * len(columns), 7.2), sharey="row", layout="constrained")
        for (ax_norms, ax_loss), (col, norms, diff) in zip(axes.T, columns):
            for k, n in norms.items():
                layer, _, kind = k.rpartition(".")
                n = np.where(np.array(n) > 0, n, np.nan)  # gaps at exact zeros (step 0)
                ax_norms.plot(steps[sel], n[sel], {"weight": "-", "bias": "--"}.get(kind, ":"), lw=1, color=colors[layer or kind], label=k)
            ax_norms.set(title=f"op norms of ({col})", xlabel="step", yscale="log" if log else "linear")
            diff = np.where(np.abs(diff) > 0, np.abs(diff), np.nan)  # gaps at exact zeros
            ax_loss.plot(steps[sel], diff[sel], "o-", ms=3, lw=1, color="C0")
            ax_loss.set(title=f"|test loss difference| ({col})", xlabel="step", yscale="log" if log else "linear")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside right center", fontsize=7, frameon=False)
        fig.suptitle(f"vmapped vs. plain training on all the data, fp32 vs. fp64 states; {title}", fontsize=10)
        fig.savefig(exp_dir / f"{name}.png", dpi=150)
    if show:
        plt.show()
    plt.close("all")


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--replot", is_flag=True, help="only redo the plots from DIR/vmap_error.json")
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dir, device, replot, show):
    if replot:
        plot(exp_dir, json.loads((exp_dir / "vmap_error.json").read_text()), Experiment(exp_dir, device).name, show)
    else:
        vmap_error(exp_dir, device, show)


if __name__ == "__main__":
    main()
