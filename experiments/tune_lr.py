"""

Train with several learning rates, pick the lowest final test loss and save it to DIR/config.json:
  python -m experiments.tune_lr DIR [--lrs 0.01,0.02,...] [--steps n] [--device cpu] [--no-show]

Also saves DIR/lr_tuning.json and DIR/lr_tuning.png.
"""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from main import Experiment
from src.utils import eval_loss


def tune_lr(exp_dir: Path, lrs: list[float], steps: int | None = None, device: str = "cpu", compile: bool = False, show: bool = True) -> float:
    exp = Experiment(exp_dir, device, compile)
    exp = replace(exp, cfg=replace(exp.cfg, lr=lrs[0]))  # the config may have no lr yet
    steps = exp.cfg.n_steps if steps is None else steps
    all_test = torch.arange(len(exp.test_ds))
    losses, accs = [], []
    for lr in tqdm(lrs, desc="learning rates"):
        init_state = replace(exp.init_state, lr=replace(exp.init_state.lr, lr=lr))
        final = exp.magic.train(init_state, n_steps=steps)
        loss, metrics, _ = eval_loss(exp.model, exp.loss_fn, {k: p.to(device) for k, p in final.params.items()}, exp.test_loader, all_test, bf16=exp.bf16)
        losses.append(loss.item() if math.isfinite(loss.item()) else math.inf)
        accs.append(metrics["acc"].item())
        click.echo(f"  lr {lr:g}: test loss {losses[-1]:.4f}, acc {accs[-1]:.3f}")
    best = lrs[min(range(len(lrs)), key=lambda i: losses[i])]
    click.echo(f"best lr: {best:g} (saved to {exp_dir / 'config.json'})")
    (exp_dir / "config.json").write_text(json.dumps(asdict(replace(exp.cfg, lr=best)), indent=2))
    (exp_dir / "lr_tuning.json").write_text(json.dumps({"steps": steps, "lrs": lrs, "test_loss": losses, "test_acc": accs, "best": best}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for ax, values, name in zip(axes, [losses, accs], ["test loss", "test acc"]):
        ax.plot(lrs, values, "o-")
        ax.axvline(best, color="gray", ls="--", lw=0.8)
        ax.set(xscale="log", xlabel="lr", ylabel=name)
        ax.set_xticks(lrs, [f"{lr:g}" for lr in lrs])
        ax.tick_params(which="minor", labelbottom=False)
    fig.suptitle(f"{exp.cfg.model} on {exp.cfg.dataset}, {steps} steps", fontsize=10)
    fig.savefig(exp_dir / "lr_tuning.png", dpi=150)
    if show:
        plt.show()
    plt.close("all")
    return best


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--lrs", type=str, default="0.01,0.02,0.05,0.1,0.2,0.5,1", show_default=True, help="comma-separated learning rates")
@click.option("--steps", type=int, default=None, help="train this many steps (default: n_steps)")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--compile/--no-compile", default=False, help="torch.compile the training passes")
@click.option("--show/--no-show", default=True, help="show the plot")
def main(exp_dir, lrs, steps, device, compile, show):
    tune_lr(exp_dir, [float(v) for v in lrs.split(",")], steps, device, compile, show)


if __name__ == "__main__":
    main()
