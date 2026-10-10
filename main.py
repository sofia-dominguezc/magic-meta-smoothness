"""
  python main.py new DIR [settings]  # create an experiment: save its settings (dataset, model, optimizer, ...) to DIR/config.json
  python main.py tune-lr DIR         # train the small model with several learning rates, save the best one to the config
  python main.py retrain DIR         # train the full model (logged) and models with perturbed training weights (data dropped / noised)
  python main.py run-magic DIR       # train on all the data and compute df/dw for each eval subset
  python main.py evaluate DIR        # compare MAGIC's predictions with the retrained models

The commands after `new` only take the experiment DIR (and --device). The learning rate may be omitted in `new`:
then tune-lr must be run first (μP: the learning rate tuned on the small model transfers to the default size).
"""

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.data import DATASETS, Dataset, EvalLoader, Loader, Perturbation, Subset
from src.magic import LOSSES, AdamState, Magic, OptimState, SGDMomentumState, SGDState, eval_loss
from src.models import MODELS, adam_lr_mults, build_model
from src.utils import Logger, agreement_metrics, eos_summary, plot_scatter_grid


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    dataset: str = "mnist"
    model: str = "cnn"
    size: str = "default"
    loss: str = "ce"
    init_seed: int = 0
    data_seed: int = 0
    batch_size: int = 32
    optimizer: str = "sgd"
    lr: float | None = None
    momentum: float | None = None  # SGD: None = no momentum; Adam: beta1 (None = 0.9)
    weight_decay: float = 0.0
    n_steps: int = 400
    ckpt_frac: float = 0.25

    @property
    def ckpt_every(self) -> int:
        every = self.n_steps * self.ckpt_frac
        assert abs(every - round(every)) < 1e-6, f"n_steps * ckpt_frac = {every} must be an integer"
        return round(every)


class Experiment:
    """An experiment dir with its saved config, datasets, and (deterministic) model + initial state."""

    def __init__(self, dir: Path, device: str):
        self.dir, self.device = dir, device
        self.cfg = ExperimentConfig(**json.loads((dir / "config.json").read_text()))
        self.loss_fn = LOSSES[self.cfg.loss]
        self.train_ds: Dataset = DATASETS[self.cfg.dataset](train=True)
        self.test_ds: Dataset = DATASETS[self.cfg.dataset](train=False)

    def build(self, size: str | None = None, lr: float | None = None) -> tuple[torch.nn.Module, OptimState]:
        """The model (of size `size`, default cfg.size) and its initial state (with lr `lr`, default cfg.lr)."""
        cfg = self.cfg
        lr = cfg.lr if lr is None else lr
        if lr is None:
            raise click.UsageError(f"no learning rate in {self.dir / 'config.json'}: run tune-lr first")
        torch.manual_seed(cfg.init_seed)
        model = build_model(cfg.model, size or cfg.size, self.train_ds.in_shape, self.train_ds.out_dim).to(self.device)
        params = {k: p.detach().clone() for k, p in model.named_parameters()}
        if cfg.optimizer == "adam":
            betas = (0.9 if cfg.momentum is None else cfg.momentum, 0.999)
            state = AdamState(params=params, lr=lr, lr_mults=adam_lr_mults(model), betas=betas, weight_decay=cfg.weight_decay)
        elif cfg.momentum is not None:
            state = SGDMomentumState(params=params, lr=lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        else:
            state = SGDState(params=params, lr=lr, weight_decay=cfg.weight_decay)
        return model, state

    def loader(self, **kwargs) -> Loader:
        return Loader(self.train_ds, self.cfg.batch_size, seed=self.cfg.data_seed, device=self.device, **kwargs)

    def magic(self, model: torch.nn.Module, logger: Logger | None = None, **kwargs) -> Magic:
        return Magic(model, self.loader(), self.loss_fn, logger, **kwargs)


def parse_labels(ctx, param, value: str) -> list[int | None]:
    return [None if v.strip().lower() in ("none", "all") else int(v) for v in value.split(",")]


def parse_floats(ctx, param, value: str) -> list[float]:
    return [float(v) for v in value.split(",")]


def run_suffix(start: int) -> str:
    """Suffix of the dirs of runs / MAGIC results that only act on the steps from `start` on."""
    return "" if start == 0 else f"_from_{start}"


exp_dir = click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
device_option = click.option("--device", type=str, default="cpu", show_default=True)
show_option = click.option("--show/--no-show", default=True, help="show the plots")


@click.group(help=__doc__)
def main():
    pass


@main.command(help="Create the experiment DIR and save its settings to DIR/config.json.")
@exp_dir
@click.option("--dataset", type=click.Choice(list(DATASETS)), default="mnist", show_default=True)
@click.option("--model", type=click.Choice(list(MODELS)), default="cnn", show_default=True)
@click.option("--size", type=click.Choice(["small", "default"]), default="default", show_default=True, help="model size (small: for tuning)")
@click.option("--loss", type=click.Choice(list(LOSSES)), default="ce", show_default=True)
@click.option("--init-seed", type=int, default=0, show_default=True)
@click.option("--data-seed", type=int, default=0, show_default=True, help="seed of the batch order")
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--optimizer", type=click.Choice(["sgd", "adam"]), default="sgd", show_default=True)
@click.option("--lr", type=float, default=None, help="learning rate (default: set by tune-lr)")
@click.option("--momentum", type=float, default=None, help="SGD momentum (default: none) or Adam's beta1 (default: 0.9)")
@click.option("--weight-decay", type=float, default=0.0, show_default=True)
@click.option("--n-steps", type=int, default=400, show_default=True)
@click.option("--ckpt-frac", type=float, default=0.25, show_default=True, help="checkpoint every this fraction of training")
def new(exp_dir: Path, **kwargs):
    cfg = ExperimentConfig(**kwargs)
    path = exp_dir / "config.json"
    if path.exists():
        raise click.UsageError(f"{path} exists")
    exp_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2))
    click.echo(f"saved {path}" + ("" if cfg.lr else " (no lr: run tune-lr)"))


@main.command("tune-lr", help="Train the small model with each lr, pick the lowest final test loss and save it to DIR/config.json.")
@exp_dir
@device_option
@click.option("--lrs", type=str, default="0.01,0.02,0.05,0.1,0.2,0.5,1,2", callback=parse_floats, show_default=True, help="comma-separated learning rates")
@click.option("--steps", type=int, default=None, help="train this many steps (default: n_steps)")
@show_option
def tune_lr(exp_dir, device, lrs, steps, show):
    exp = Experiment(exp_dir, device)
    steps = exp.cfg.n_steps if steps is None else steps
    test_loader = EvalLoader(exp.test_ds, device=device)
    losses, accs = [], []
    for lr in tqdm(lrs, desc="learning rates"):
        model, init_state = exp.build(size="small", lr=lr)
        final, _ = exp.magic(model).forward_pass(init_state, n_steps=steps)
        loss, metrics, _ = eval_loss(model, exp.loss_fn, {k: p.to(device) for k, p in final.params.items()}, test_loader)
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
    fig.suptitle(f"{exp.cfg.model} (small) on {exp.cfg.dataset}, {steps} steps", fontsize=10)
    fig.savefig(exp_dir / "lr_tuning.png", dpi=150)
    if show:
        plt.show()
    plt.close("all")


@main.command(help="Train the full model (logged, into DIR/model_states) and one model per (label, seed) with the training "
              "weights perturbed from step --start on: dropping a random --alpha fraction of the data (--kind drop), or "
              "adding alpha * N(0, 1) to every weight (--kind randn).")
@exp_dir
@device_option
@click.option("--kind", type=click.Choice(["drop", "randn"]), default="drop", show_default=True)
@click.option("--alpha", type=float, default=0.05, show_default=True, help="size of the perturbation")
@click.option("--labels", type=str, default="all", callback=parse_labels, help="drop only: comma-separated labels to drop from, 'all' = any class")
@click.option("--start", type=int, default=0, show_default=True, help="perturb the weights from this step on (trained on all the data before)")
@click.option("--n-seeds", type=int, default=10, show_default=True)
@click.option("--seed0", type=int, default=0, show_default=True, help="perturbation seeds are seed0, ..., seed0 + n_seeds - 1")
@click.option("--full/--no-full", default=True, help="also train the full model")
@click.option("--num-top-eigs", type=int, default=1, show_default=True, help="full model only: log the top Hessian eigenvalues (0: don't)")
@click.option("--log-tangent/--no-log-tangent", default=True, help="full model only: log the norms of ds_t/ds_0 v for a random v (forward-mode AD)")
@click.option("--eval-every", type=int, default=100, show_default=True, help="full model only: log the test loss and metrics every this many steps (0: never)")
@click.option("--power-iters", type=int, default=3, show_default=True, help="full model only: power iterations per step for the operator norms")
@click.option("--wandb", "wandb_project", type=str, default=None, help="full model only: wandb project to log to")
@show_option
def retrain(exp_dir, device, kind, alpha, labels, start, n_seeds, seed0, full, num_top_eigs, log_tangent, eval_every, power_iters, wandb_project, show):
    """Only the full model is logged; the retrained models only save their checkpoints."""
    exp = Experiment(exp_dir, device)
    cfg = exp.cfg
    model, init_state = exp.build()
    labels = [None] if kind == "randn" else labels
    perturbations = [Perturbation(kind=kind, alpha=alpha, label=label, seed=seed0 + s) for label in labels for s in range(n_seeds)]

    def train_and_save(run_dir: Path, perturbation: Perturbation | None, logger: Logger | None = None):
        """Train with the weights of `perturbation` (None: all ones), logging if a logger is given."""
        run_dir.mkdir(parents=True, exist_ok=True)
        weights = None if perturbation is None else perturbation.weights(exp.train_ds)
        magic = Magic(
            model, exp.loader(weights=weights, weights_from=start), exp.loss_fn, logger,
            test_loader=EvalLoader(exp.test_ds, device=device) if eval_every else None,
            eval_every=eval_every, num_top_eigs=num_top_eigs, log_tangent=log_tangent,
        )
        final, ckpts = magic.forward_pass(init_state, n_steps=cfg.n_steps, ckpt_every=cfg.ckpt_every, log=logger is not None)
        for j, state in enumerate([*ckpts, final]):
            ckpt_num = min(j * cfg.ckpt_every, cfg.n_steps)
            torch.save(asdict(state), run_dir / f"ckpt_{ckpt_num}.pt")
        if perturbation is not None:
            (run_dir / "perturbation.json").write_text(json.dumps({**asdict(perturbation), "start": start}))

    if full:
        click.echo("training the full model")
        name = f"{cfg.model} ({cfg.size}) on {cfg.dataset}, {cfg.loss}, {cfg.optimizer} lr {cfg.lr:g}"
        logger = Logger(wandb_project, name, asdict(cfg), power_iters=power_iters)
        train_and_save(exp_dir / "model_states", None, logger)
        logger.finish(exp_dir / "model_states" / "metrics.pt")
        if num_top_eigs:
            eig = torch.load(exp_dir / "model_states" / "metrics.pt", weights_only=True)["eig/1"]["value"].numpy()
            eos = eos_summary(eig)
            (exp_dir / "model_states" / "eos.json").write_text(json.dumps(eos, indent=2))
            click.echo("edge of stability: " + ("not reached" if eos["eos_step"] is None else
                       f"reached at step {eos['eos_step']}, then {eos['eos_frac']:.0%} of the steps above it") + f" (max EMA {eos['max_ema']:.2f})")
        if show:
            plt.show()
        plt.close("all")
    for p in tqdm(perturbations, desc="retrained models"):
        train_and_save(exp_dir / "runs" / f"{p.name}{run_suffix(start)}", p)


@main.command("run-magic", help="Train on all the data and compute df/dw (MAGIC) for the loss on each eval subset at step `--step`, "
              "for the weights of the steps [--start, --step) (any steps; `evaluate` needs retrained models with the same --start "
              "and a checkpoint at --step).")
@exp_dir
@device_option
@click.option("--step", type=int, default=None, help="the last training step (default: n_steps)")
@click.option("--start", type=int, default=0, show_default=True, help="the first step whose weights are differentiated")
@click.option("--labels", type=str, default="all", callback=parse_labels, help="comma-separated labels, 'all' = any class")
@click.option("--frac", type=float, default=0.1, show_default=True, help="fraction of the test set in each eval subset (all from one label for label subsets)")
@click.option("--n-seeds", type=int, default=1, show_default=True)
@click.option("--seed0", type=int, default=0, show_default=True, help="subset seeds are seed0, ..., seed0 + n_seeds - 1")
@click.option("--passes", type=int, default=2, show_default=True, help="re-trainings to reconstruct the states in reverse (memory: passes * step^(1/passes) states)")
@click.option("--power-iters", type=int, default=3, show_default=True, help="power iterations per step for the operator norms")
@click.option("--wandb", "wandb_project", type=str, default=None, help="wandb project to log the norms of df/ds_t to")
@show_option
def run_magic(exp_dir, device, step, start, labels, frac, n_seeds, seed0, passes, power_iters, wandb_project, show):
    exp = Experiment(exp_dir, device)
    cfg = exp.cfg
    model, init_state = exp.build()
    step = cfg.n_steps if step is None else step
    magic_dir = exp_dir / "magic" / f"step_{step}{run_suffix(start)}"
    magic_dir.mkdir(parents=True, exist_ok=True)
    if start > 0:  # the state at `start`: the full model's checkpoint if there is one, else train to it
        ckpt = exp_dir / "model_states" / f"ckpt_{start}.pt"
        if ckpt.exists():
            init_state = type(init_state)(**torch.load(ckpt, weights_only=True))
        else:
            init_state, _ = exp.magic(model).forward_pass(init_state, n_steps=start)

    subsets = [Subset(frac=frac, label=label, seed=seed0 + s) for label in labels for s in range(n_seeds)]
    eval_loaders = {
        f"{'test set' if s.label is None else f'test label {s.label}'} (seed {s.seed})":
        EvalLoader(exp.test_ds, s.mask(exp.test_ds).nonzero()[:, 0], device=device)
        for s in subsets
    }
    name = f"MAGIC steps {start}-{step}: norms of df/ds_t, f = loss on {frac:.0%} of the test set"
    logger = Logger(wandb_project, name, {**asdict(cfg), "start": start, "step": step}, power_iters=power_iters)
    outputs, betas = exp.magic(model, logger).backward_pass(
        init_state, n_steps=step - start, start=start, eval_loaders=eval_loaders, passes=passes
    )
    logger.finish(magic_dir / "metrics.pt")
    if show:
        plt.show()
    plt.close("all")

    outputs_cpu, betas_cpu = torch.stack(list(outputs.values())).cpu(), torch.stack(list(betas.values())).cpu()  # [n_fns], [n_fns, N]
    torch.save({"outputs": outputs_cpu, "betas": betas_cpu, "start": start, "step": step}, magic_dir / "linearization.pt")
    (magic_dir / "eval_subsets.json").write_text(json.dumps([asdict(s) for s in subsets], indent=2))
    for subset, out in zip(subsets, outputs_cpu.tolist()):
        click.echo(f"  {subset}: f = {out:.4f}")


@main.command(help="Compare MAGIC's linear predictions with the retrained models at step `--step` (both with `--start`).")
@exp_dir
@device_option
@click.option("--step", type=int, default=None, help="the step given to run-magic (default: n_steps)")
@click.option("--start", type=int, default=0, show_default=True, help="the start given to run-magic and retrain")
@show_option
def evaluate(exp_dir, device, step, start, show):
    exp = Experiment(exp_dir, device)
    cfg = exp.cfg
    model, _ = exp.build()
    step = cfg.n_steps if step is None else step
    magic_dir = exp_dir / "magic" / f"step_{step}{run_suffix(start)}"

    lin = torch.load(magic_dir / "linearization.pt", weights_only=True)
    outputs, betas = lin["outputs"], lin["betas"]
    subsets = [Subset(**d) for d in json.loads((magic_dir / "eval_subsets.json").read_text())]
    eval_loaders = [EvalLoader(exp.test_ds, s.mask(exp.test_ds).nonzero()[:, 0], device=device) for s in subsets]

    run_dirs = [d for d in (exp_dir / "runs").glob("*") if (d / f"ckpt_{step}.pt").exists()]
    runs = {d: json.loads((d / "perturbation.json").read_text()) for d in run_dirs}
    run_dirs = [d for d in run_dirs if runs[d].pop("start", 0) == start]
    runs = {d: Perturbation(**runs[d]) for d in run_dirs}
    run_dirs.sort(key=lambda d: (runs[d].group, runs[d].seed))
    if not run_dirs:
        raise click.UsageError(f"no retrained runs from step {start} with a checkpoint at step {step} in {exp_dir / 'runs'}")
    click.echo(f"evaluating {len(subsets)} eval subsets on {len(run_dirs)} retrained models at step {step} (perturbed from step {start})")
    preds, reals, groups = [], [], []
    for run_dir in run_dirs:
        params = torch.load(run_dir / f"ckpt_{step}.pt", weights_only=True)["params"]
        params = {k: p.to(device) for k, p in params.items()}
        preds.append(outputs + betas @ (runs[run_dir].weights(exp.train_ds) - 1).to(betas.dtype))  # f(w) ~ f(1) + betas @ (w - 1)
        reals.append(torch.stack([eval_loss(model, exp.loss_fn, params, loader)[0] for loader in eval_loaders]).cpu())
        groups.append(runs[run_dir].group)
    preds_np, reals_np = torch.stack(preds, 1).numpy(), torch.stack(reals, 1).numpy()

    # one row per test label, one column per seed
    rows = list(dict.fromkeys(s.label for s in subsets))
    cols = sorted(set(s.seed for s in subsets))
    positions = [(rows.index(s.label), cols.index(s.seed)) for s in subsets]
    titles = [
        f"eval loss on {s.frac:.0%} of the test set{'' if s.label is None else f', from label {s.label}'} (seed {s.seed})"
        for s in subsets
    ]
    title = f"MAGIC vs. retraining, steps {start}-{step} of {cfg.n_steps}: {len(run_dirs)} runs with the train weights perturbed from step {start}"
    eval_dir = exp_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    name = f"step_{step}{run_suffix(start)}"
    metrics = plot_scatter_grid(preds_np, reals_np, outputs.numpy(), positions, titles, groups, eval_dir / f"{name}.png", title)
    full = outputs.numpy()[:, None]
    pooled = agreement_metrics((preds_np - full).ravel(), (reals_np - full).ravel())  # all eval fns together, centered
    results = {
        "step": step, "start": start, "runs": [d.name for d in run_dirs], "eval_subsets": [asdict(s) for s in subsets],
        "metrics": metrics, "pooled": pooled, "min_ccc": min(m["ccc"] for m in metrics),
        "full": outputs.tolist(), "preds": preds_np.tolist(), "reals": reals_np.tolist(),
    }
    (eval_dir / f"{name}.json").write_text(json.dumps(results, indent=2))
    for subset, m in zip(subsets, metrics):
        click.echo(f"  {subset}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
    click.echo(f"  pooled: " + ", ".join(f"{k}={v:.3f}" for k, v in pooled.items()) + f"; min ccc={results['min_ccc']:.3f}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
