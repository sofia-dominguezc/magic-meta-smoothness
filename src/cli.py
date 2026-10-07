"""
  python main.py train-subsets  # train models on random training subsets (dropping `frac` of the data)
  python main.py run-magic      # train on all the data and compute df/dw for each eval subset
  python main.py evaluate       # compare MAGIC's predictions with the retrained models

All three share the experiment options (dataset, model, seeds, optimizer, ...), which
are saved to <out-dir>/config.json and must agree between commands.
"""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.data import DATASETS, Dataset, EvalLoader, Loader, Subset
from src.magic import LOSSES, OptimState, SGDMomentumState, SGDState, eval_loss, magic, train
from src.models import CNN, MLP
from src.utils import Logger, plot_scatter_grid


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    dataset: str = "majority"
    loss: str = "mse"
    model: str = "mlp"
    width: int = 32
    depth: int = 1
    init_seed: int = 0
    data_seed: int = 0
    batch_size: int = 32
    lr: float = 0.01
    momentum: float | None = None
    weight_decay: float = 0.0
    n_steps: int = 400
    ckpt_frac: float = 0.25
    drop_frac: float = 0.05  # for train data

    @property
    def ckpt_every(self) -> int:
        every = self.n_steps * self.ckpt_frac
        assert abs(every - round(every)) < 1e-6, f"n_steps * ckpt_frac = {every} must be an integer"
        return round(every)


_CONFIG_OPTIONS = [
    click.option("--dataset", type=click.Choice(list(DATASETS)), default="majority", show_default=True),
    click.option("--loss", type=click.Choice(list(LOSSES)), default="mse", show_default=True),
    click.option("--model", type=click.Choice(["mlp", "cnn"]), default="mlp", show_default=True),
    click.option("--width", type=int, default=32, show_default=True),
    click.option("--depth", type=int, default=1, show_default=True),
    click.option("--init-seed", type=int, default=0, show_default=True),
    click.option("--data-seed", type=int, default=0, show_default=True, help="seed of the batch order"),
    click.option("--batch-size", type=int, default=32, show_default=True),
    click.option("--lr", type=float, default=0.01, show_default=True),
    click.option("--momentum", type=float, default=None, show_default=True),
    click.option("--weight-decay", type=float, default=0.0, show_default=True),
    click.option("--n-steps", type=int, default=400, show_default=True),
    click.option("--ckpt-frac", type=float, default=0.25, show_default=True, help="checkpoint every this fraction of training"),
    click.option("--drop-frac", type=float, default=0.05, show_default=True, help="fraction of the (label's) training examples each retrained model drops"),
    click.option("--out-dir", type=click.Path(path_type=Path), default=Path("./results"), show_default=True),
    click.option("--device", type=str, default="cpu", show_default=True),
]


def config_options(f):
    for option in reversed(_CONFIG_OPTIONS):
        f = option(f)
    return f


def parse_labels(ctx, param, value: str) -> list[int | None]:
    return [None if v.strip().lower() in ("none", "all") else int(v) for v in value.split(",")]


def setup(kwargs: dict) -> tuple[ExperimentConfig, Path, str, Dataset, Dataset, torch.nn.Module, OptimState]:
    """Read the experiment config from the click kwargs (checked against the one saved in
    out_dir), and build the datasets, the model and the (deterministic) initial state."""
    cfg = ExperimentConfig(**{f.name: kwargs.pop(f.name) for f in fields(ExperimentConfig)})
    out_dir, device = kwargs.pop("out_dir"), kwargs.pop("device")
    path = out_dir / "config.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != asdict(cfg):
            diff = {k: (saved.get(k), v) for k, v in asdict(cfg).items() if saved.get(k) != v}
            raise click.UsageError(f"config differs from {path} (saved, given): {diff}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(cfg), indent=2))

    train_ds = DATASETS[cfg.dataset](train=True)
    test_ds = DATASETS[cfg.dataset](train=False)
    x, _ = train_ds[0]
    torch.manual_seed(cfg.init_seed)
    if cfg.model == "mlp":
        model = MLP(x.numel(), train_ds.out_dim, cfg.width, cfg.depth)
    else:
        model = CNN(x.shape[0], train_ds.out_dim, cfg.width, cfg.depth)
    model = model.to(device)
    params = {k: p.detach().clone() for k, p in model.named_parameters()}
    if cfg.momentum is not None:
        state = SGDMomentumState(params=params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
    else:
        state = SGDState(params=params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return cfg, out_dir, device, train_ds, test_ds, model, state


@click.group(help=__doc__)
def main():
    pass


@main.command(help="Train one model per (label, seed), each dropping a random `drop_frac` of the training data.")
@config_options
@click.option("--labels", type=str, default="all", callback=parse_labels, help="comma-separated labels, 'all' = any class")
@click.option("--n-seeds", type=int, default=10, show_default=True)
@click.option("--seed0", type=int, default=0, show_default=True, help="subset seeds are seed0, ..., seed0 + n_seeds - 1")
@click.option("--full/--no-full", default=True, help="also train the full model (no data dropped) into <out-dir>/model_states, with logging")
@click.option("--num-top-eigs", type=int, default=0, show_default=True, help="full model only: log the top Hessian eigenvalues (slow)")
@click.option("--eval-every", type=int, default=None, help="full model only: log the test loss and metrics every this many steps (default: ckpt_every, 0: never)")
@click.option("--wandb", "wandb_project", type=str, default=None, help="full model only: wandb project to log to")
@click.option("--show/--no-show", default=True, help="full model only: show the training plots")
def train_subsets(labels, n_seeds, seed0, full, num_top_eigs, eval_every, wandb_project, show, **kwargs):
    """Only the full model is logged; the retrained models only save their checkpoints."""
    if not full and (num_top_eigs or eval_every or wandb_project):
        raise click.UsageError("--num-top-eigs, --eval-every and --wandb are only for the full model, so they need --full")
    cfg, out_dir, device, train_ds, test_ds, model, init_state = setup(kwargs)
    eval_every = cfg.ckpt_every if eval_every is None else eval_every
    subsets = [Subset(frac=cfg.drop_frac, label=label, seed=seed0 + s) for label in labels for s in range(n_seeds)]

    def train_and_save(run_dir: Path, subset: Subset | None, logger: Logger | None = None):
        """Train dropping `subset` (None: no data dropped), with a per-step progress bar if logged."""
        run_dir.mkdir(parents=True, exist_ok=True)
        weights = None if subset is None else (~subset.mask(train_ds)).float()
        loader = Loader(train_ds, cfg.batch_size, seed=cfg.data_seed, weights=weights, device=device)
        final, ckpts = train(
            model, loader, LOSSES[cfg.loss], init_state,
            n_steps=cfg.n_steps, ckpt_every=cfg.ckpt_every, logger=logger,
            eval_loader=EvalLoader(test_ds, device=device) if logger is not None and eval_every else None,
            eval_every=eval_every, num_top_eigs=num_top_eigs if logger is not None else 0,
            progress=logger is not None,
        )
        for j, state in enumerate([*ckpts, final]):
            ckpt_num = min(j * cfg.ckpt_every, cfg.n_steps)
            torch.save(asdict(state.apply(lambda t: t.cpu())), run_dir / f"ckpt_{ckpt_num}.pt")
        if subset is not None:
            (run_dir / "subset.json").write_text(json.dumps(asdict(subset)))

    if full:
        click.echo("training the full model")
        logger = Logger(wandb_project, "full model", asdict(cfg))
        train_and_save(out_dir / "model_states", None, logger)
        logger.finish(out_dir / "model_states" / "metrics.pt")
        if show:
            plt.show()
        plt.close("all")
    for s in tqdm(subsets, desc="retrained models"):
        dropped = "all_labels" if s.label is None else f"label_{s.label}"
        train_and_save(out_dir / "runs" / f"drop_{dropped}_seed_{s.seed}", s)


@main.command(help="Train on all the data and compute df/dw (MAGIC) for the loss on each eval subset, at step `--step`.")
@config_options
@click.option("--step", type=int, default=None, help="train for this many steps instead of n_steps (should be a checkpoint step)")
@click.option("--labels", type=str, default="all", callback=parse_labels, help="comma-separated labels, 'all' = any class")
@click.option("--frac", type=float, default=0.1, show_default=True, help="fraction of the test set (or label's examples) in each eval subset")
@click.option("--n-seeds", type=int, default=1, show_default=True)
@click.option("--seed0", type=int, default=0, show_default=True, help="subset seeds are seed0, ..., seed0 + n_seeds - 1")
@click.option("--show/--no-show", default=True)
def run_magic(step, labels, frac, n_seeds, seed0, show, **kwargs):
    cfg, out_dir, device, train_ds, test_ds, model, init_state = setup(kwargs)
    step = cfg.n_steps if step is None else step
    magic_dir = out_dir / "magic" / f"step_{step}"
    magic_dir.mkdir(parents=True, exist_ok=True)

    subsets = [Subset(frac=frac, label=label, seed=seed0 + s) for label in labels for s in range(n_seeds)]
    eval_loaders = [EvalLoader(test_ds, s.mask(test_ds).nonzero()[:, 0], device=device) for s in subsets]
    test_names = ["test set" if s.label is None else f"test label {s.label}" for s in subsets]
    logger = Logger(name=f"MAGIC at step {step}, train drop {cfg.drop_frac:.0%}")
    logger.labels["delta"] = test_names  # one color per test label
    logger.ncols["betas"] = n_seeds  # one row per test label, one column per seed
    logger.ylabels["delta"] = "RMS of df/ds_t"
    outputs, betas = magic(
        model, Loader(train_ds, cfg.batch_size, seed=cfg.data_seed, device=device), LOSSES[cfg.loss], init_state, step,
        eval_loaders, logger=logger, eval_names=[f"{name}, seed {s.seed}" for name, s in zip(test_names, subsets)],
    )
    logger.finish(magic_dir / "metrics.pt", separate={
        "delta": f"eval loss on {n_seeds} different {frac:.0%} subsets of each test label",
        "betas": f"eval loss on {frac:.0%} of each test label (numbers: train example index)",
    })
    if show:
        plt.show()
    plt.close("all")

    outputs_cpu, betas_cpu = torch.stack(outputs).cpu(), torch.stack(betas).cpu()  # [n_fns], [n_fns, N]
    torch.save({"outputs": outputs_cpu, "betas": betas_cpu}, magic_dir / "linearization.pt")
    (magic_dir / "eval_subsets.json").write_text(json.dumps([asdict(s) for s in subsets], indent=2))
    for subset, out in zip(subsets, outputs_cpu.tolist()):
        click.echo(f"  {subset}: f = {out:.4f}")


@main.command(help="Compare MAGIC's linear predictions with the retrained subset models at step `--step`.")
@config_options
@click.option("--step", type=int, default=None, help="the step given to run-magic (default: n_steps)")
@click.option("--show/--no-show", default=True)
def evaluate(step, show, **kwargs):
    cfg, out_dir, device, train_ds, test_ds, model, _ = setup(kwargs)
    step = cfg.n_steps if step is None else step
    magic_dir = out_dir / "magic" / f"step_{step}"

    lin = torch.load(magic_dir / "linearization.pt", weights_only=True)
    outputs, betas = lin["outputs"], lin["betas"]
    subsets = [Subset(**d) for d in json.loads((magic_dir / "eval_subsets.json").read_text())]
    eval_loaders = [EvalLoader(test_ds, s.mask(test_ds).nonzero()[:, 0], device=device) for s in subsets]

    run_dirs = [d for d in (out_dir / "runs").glob("*") if (d / f"ckpt_{step}.pt").exists()]
    run_subsets = {d: Subset(**json.loads((d / "subset.json").read_text())) for d in run_dirs}
    run_dirs.sort(key=lambda d: (run_subsets[d].label is not None, run_subsets[d].label or 0, run_subsets[d].seed))  # "all" first
    if not run_dirs:
        raise click.UsageError(f"no retrained runs with a checkpoint at step {step} in {out_dir / 'runs'}")
    click.echo(f"evaluating {len(subsets)} eval subsets on {len(run_dirs)} retrained models at step {step}")
    preds, reals, groups = [], [], []
    for run_dir in run_dirs:
        dropped = run_subsets[run_dir]
        params = torch.load(run_dir / f"ckpt_{step}.pt", weights_only=True)["params"]
        params = {k: p.to(device) for k, p in params.items()}
        # f(w) ~ f(1) + betas @ (w - 1), with w = 0 on the dropped examples
        preds.append(outputs - betas[:, dropped.mask(train_ds)].sum(1))
        reals.append(torch.stack([eval_loss(model, LOSSES[cfg.loss], params, loader)[0] for loader in eval_loaders]).cpu())
        groups.append("all labels" if dropped.label is None else f"label {dropped.label}")
    preds_np, reals_np = torch.stack(preds, 1).numpy(), torch.stack(reals, 1).numpy()

    # one row per test label, one column per seed
    rows = list(dict.fromkeys(s.label for s in subsets))
    cols = sorted(set(s.seed for s in subsets))
    positions = [(rows.index(s.label), cols.index(s.seed)) for s in subsets]
    titles = [
        f"eval loss on {s.frac:.0%} of {'the test set' if s.label is None else f'test label {s.label}'} (seed {s.seed})"
        for s in subsets
    ]
    title = f"MAGIC vs. retraining, step {step}/{cfg.n_steps}: each point drops {cfg.drop_frac:.0%} of the train data ({len(run_dirs)} runs)"
    eval_dir = out_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    metrics = plot_scatter_grid(
        preds_np, reals_np, outputs.numpy(), positions, titles, groups, eval_dir / f"step_{step}.png", title
    )
    results = {
        "step": step, "runs": [d.name for d in run_dirs], "eval_subsets": [asdict(s) for s in subsets],
        "metrics": metrics, "full": outputs.tolist(), "preds": preds_np.tolist(), "reals": reals_np.tolist(),
    }
    (eval_dir / f"step_{step}.json").write_text(json.dumps(results, indent=2))
    for subset, m in zip(subsets, metrics):
        click.echo(f"  {subset}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
    if show:
        plt.show()


if __name__ == "__main__":
    main()
