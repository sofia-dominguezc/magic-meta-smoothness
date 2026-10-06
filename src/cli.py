"""
  python cli.py train-subsets  # train models on random training subsets (dropping `frac` of the data)
  python cli.py run-magic      # train on all the data and compute df/dw for each eval subset
  python cli.py evaluate       # compare MAGIC's predictions with the retrained models

All three share the experiment options (dataset, model, seeds, optimizer, ...), which
are saved to <out-dir>/config.json and must agree between commands.
"""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch

from src.data import DATASETS, Dataset, Subset
from src.magic import LOSSES, Loader, OptimState, SGDMomentumState, SGDState, magic, subset_loss, train
from src.models import CNN, MLP
from src.utils import Logger, plot_scatter_grid

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    momentum: float = 0.0
    weight_decay: float = 0.0
    n_steps: int = 400
    ckpt_frac: float = 0.1

    @property
    def ckpt_every(self) -> int:
        every = self.n_steps * self.ckpt_frac
        assert every == int(every), f"n_steps * ckpt_frac = {every} must be an integer"
        return int(every)


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
    click.option("--momentum", type=float, default=0.0, show_default=True),
    click.option("--weight-decay", type=float, default=0.0, show_default=True),
    click.option("--n-steps", type=int, default=400, show_default=True),
    click.option("--ckpt-frac", type=float, default=0.1, show_default=True, help="checkpoint every this fraction of training"),
    click.option("--out-dir", type=click.Path(path_type=Path), default=Path("./experiments"), show_default=True),
    click.option("--device", type=str, default=DEFAULT_DEVICE, show_default=True),
]


def config_options(f):
    for option in reversed(_CONFIG_OPTIONS):
        f = option(f)
    return f


def parse_labels(ctx, param, value: str) -> list[int | None]:
    return [None if v.strip().lower() in ("none", "all") else int(v) for v in value.split(",")]


def setup(kwargs: dict) -> tuple[ExperimentConfig, Path, Dataset, Dataset, torch.nn.Module, OptimState]:
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

    train_ds = DATASETS[cfg.dataset](train=True, device=device)
    test_ds = DATASETS[cfg.dataset](train=False, device=device)
    x = train_ds.tensors["x"]
    torch.manual_seed(cfg.init_seed)
    if cfg.model == "mlp":
        model = MLP(x[0].numel(), train_ds.out_dim, cfg.width, cfg.depth)
    else:
        model = CNN(x.shape[1], train_ds.out_dim, cfg.width, cfg.depth)
    model = model.to(device)
    params = {k: p.detach().clone() for k, p in model.named_parameters()}
    if cfg.momentum:
        state = SGDMomentumState(params=params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
    else:
        state = SGDState(params=params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return cfg, out_dir, train_ds, test_ds, model, state


@click.group(help=__doc__)
def cli():
    pass


@cli.command(help="Train one model per (label, seed), each dropping a random `frac` of the training data.")
@config_options
@click.option("--labels", type=str, default="all", callback=parse_labels, help="comma-separated labels, 'all' = any class")
@click.option("--frac", type=float, default=0.05, show_default=True, help="fraction of the (label's) examples to drop")
@click.option("--n-seeds", type=int, default=10, show_default=True)
@click.option("--seed0", type=int, default=0, show_default=True, help="subset seeds are seed0, ..., seed0 + n_seeds - 1")
@click.option("--num-top-eigs", type=int, default=0, show_default=True, help="log the top Hessian eigenvalues (slow)")
@click.option("--wandb", "wandb_project", type=str, default=None, help="wandb project to log to")
@click.option("--show/--no-show", default=True)
def train_subsets(labels, frac, n_seeds, seed0, num_top_eigs, wandb_project, show, **kwargs):
    cfg, out_dir, train_ds, _, model, init_state = setup(kwargs)
    subsets = [Subset(frac=frac, label=label, seed=seed0 + s) for label in labels for s in range(n_seeds)]

    for i, subset in enumerate(subsets):
        name = f"{'all' if subset.label is None else subset.label}_{subset.frac}_{subset.seed}"
        run_dir = out_dir / "runs" / name
        run_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"[{i + 1}/{len(subsets)}] training {subset}")
        loader = Loader(train_ds, cfg.batch_size, cfg.data_seed, weights=(~subset.mask(train_ds)).float())
        logger = Logger(wandb_project, str(subset), asdict(cfg))
        final, ckpts = train(
            model, loader, LOSSES[cfg.loss], init_state,
            n_steps=cfg.n_steps, ckpt_every=cfg.ckpt_every, logger=logger, num_top_eigs=num_top_eigs,
        )
        for j, state in enumerate([*ckpts, final]):
            torch.save(asdict(state.apply(lambda t: t.cpu())), run_dir / f"ckpt_{j * cfg.ckpt_every}.pt")
        (run_dir / "subset.json").write_text(json.dumps(asdict(subset)))
        logger.finish(run_dir / "metrics.pt")
        if not show:
            plt.close("all")
    if show:
        plt.show()


@cli.command(help="Train on all the data and compute df/dw (MAGIC) for the loss on each eval subset, at step `--step`.")
@config_options
@click.option("--step", type=int, default=None, help="train for this many steps instead of n_steps (should be a checkpoint step)")
@click.option("--eval-labels", type=str, default="all", callback=parse_labels)
@click.option("--eval-frac", type=float, default=0.1, show_default=True, help="fraction of the test set (or label) in each eval subset")
@click.option("--eval-seeds", type=int, default=1, show_default=True)
@click.option("--eval-seed0", type=int, default=100, show_default=True)
def run_magic(step, eval_labels, eval_frac, eval_seeds, eval_seed0, **kwargs):
    cfg, out_dir, train_ds, test_ds, model, init_state = setup(kwargs)
    step = cfg.n_steps if step is None else step
    magic_dir = out_dir / "magic" / f"step_{step}"
    magic_dir.mkdir(parents=True, exist_ok=True)

    subsets = [
        Subset(frac=eval_frac, label=label, seed=eval_seed0 + s) for label in eval_labels for s in range(eval_seeds)
    ]
    eval_idxs = [s.mask(test_ds).nonzero()[:, 0].to(test_ds.device) for s in subsets]
    final, outputs, betas = magic(
        model, Loader(train_ds, cfg.batch_size, cfg.data_seed), LOSSES[cfg.loss], init_state, step,
        test_ds, eval_idxs,
    )

    outputs_cpu, betas_cpu = torch.stack(outputs).cpu(), torch.stack(betas).cpu()  # [n_fns], [n_fns, N]
    torch.save(asdict(final.apply(lambda t: t.cpu())), magic_dir / f"ckpt_{step}.pt")
    torch.save({"outputs": outputs_cpu, "betas": betas_cpu}, magic_dir / "linearization.pt")
    (magic_dir / "eval_subsets.json").write_text(json.dumps([asdict(s) for s in subsets], indent=2))
    for subset, out in zip(subsets, outputs_cpu.tolist()):
        click.echo(f"  {subset}: f = {out:.4f}")


@cli.command(help="Compare MAGIC's linear predictions with the retrained subset models at step `--step`.")
@config_options
@click.option("--step", type=int, default=None)
@click.option("--show/--no-show", default=True)
def evaluate(step, show, **kwargs):
    cfg, out_dir, train_ds, test_ds, model, _ = setup(kwargs)
    step = cfg.n_steps if step is None else step
    magic_dir = out_dir / "magic" / f"step_{step}"

    lin = torch.load(magic_dir / "linearization.pt", weights_only=True)
    outputs, betas = lin["outputs"], lin["betas"]
    subsets = [Subset(**d) for d in json.loads((magic_dir / "eval_subsets.json").read_text())]
    eval_idxs = [s.mask(test_ds).nonzero()[:, 0].to(test_ds.device) for s in subsets]

    run_dirs = sorted(d for d in (out_dir / "runs").iterdir() if (d / f"ckpt_{step}.pt").exists())
    click.echo(f"evaluating {len(subsets)} eval subsets on {len(run_dirs)} retrained models at step {step}")
    preds, reals = [], []
    for run_dir in run_dirs:
        dropped = Subset(**json.loads((run_dir / "subset.json").read_text())).mask(train_ds)
        params = torch.load(run_dir / f"ckpt_{step}.pt", weights_only=True)["params"]
        params = {k: p.to(test_ds.device) for k, p in params.items()}
        preds.append(-betas[:, dropped].sum(1))  # betas @ (w - 1), with w = 0 on the dropped examples
        losses = [subset_loss(model, LOSSES[cfg.loss], params, test_ds, idx)[0] for idx in eval_idxs]
        reals.append(torch.stack(losses).cpu() - outputs)
    preds_np, reals_np = torch.stack(preds, 1).numpy(), torch.stack(reals, 1).numpy()

    eval_dir = out_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    metrics = plot_scatter_grid(preds_np, reals_np, [str(s) for s in subsets], eval_dir / f"step_{step}.png", f"step {step}")
    results = {
        "step": step, "runs": [d.name for d in run_dirs], "eval_subsets": [asdict(s) for s in subsets],
        "metrics": metrics, "preds": preds_np.tolist(), "reals": reals_np.tolist(),
    }
    (eval_dir / f"step_{step}.json").write_text(json.dumps(results, indent=2))
    for subset, m in zip(subsets, metrics):
        click.echo(f"  {subset}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
    if show:
        plt.show()


if __name__ == "__main__":
    cli()
