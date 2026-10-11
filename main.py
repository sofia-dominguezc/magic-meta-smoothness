"""

  python main.py new DIR [settings]   # create an experiment: save its settings (dataset, model, optimizer, start, ...) to DIR/config.json
  python main.py retrain DIR [...]    # train the full model (logged) and models with perturbed training weights (data dropped / noised)
  python main.py run-magic DIR [...]  # train on all the data and compute df/dw for each eval subset


Everything saved in DIR is comparable (same config, so same training and same steps `start`-`n_steps` perturbed):
  DIR/full_model/   final.pt (the state after training on all the data), metrics.pt + plots (the logged training)
  DIR/retrained/    one dir per perturbation (kind, alpha, label, index): final.pt, perturbation.json
  DIR/magic/        one dir per eval subset (frac, label, index): linearization.pt (f and df/dw), subset.json;
                    metrics.pt + plots (the logged df/ds_t norms) of the last run-magic call

Both commands can be run several times (e.g. with other alphas, more subsets): existing models / subsets are skipped.
experiments/ compares them (experiments/evaluate.py), tunes the lr (experiments/tune_lr.py; needed if `new` is given
no --lr) and runs whole sweeps (experiments/sweep.py).
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.data import DATASETS, Dataset, EvalLoader, Loader, Perturbation, Subset
from src.magic import LOSSES, AdamState, Magic, OptimState, SGDMomentumState, SGDState
from src.models import MODELS, adam_lr_mults
from src.utils import Logger


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    dataset: str = "mnist"
    model: str = "cnn"
    loss: str = "ce"
    init_seed: int = 0
    data_seed: int = 0
    batch_size: int = 32
    optimizer: str = "sgd"
    lr: float | None = None
    momentum: float | None = None  # SGD: None = no momentum; Adam: beta1 (None = 0.9)
    weight_decay: float = 0.0
    n_steps: int = 400
    start: int = 0  # the training weights are perturbed / differentiated from this step on
    dtype: str = "float32"  # float32, float64, or bf16 (mixed precision: float32 parameters, bfloat16 model forward)


class Experiment:
    """An experiment dir with its saved config, datasets, and (deterministic) model + initial state."""

    def __init__(self, dir: Path, device: str):
        self.dir, self.device = dir, device
        self.cfg = ExperimentConfig(**json.loads((dir / "config.json").read_text()))
        torch.set_default_dtype(torch.float64 if self.cfg.dtype == "float64" else torch.float32)
        self.loss_fn = LOSSES[self.cfg.loss]
        self.train_ds: Dataset = DATASETS[self.cfg.dataset](train=True)
        self.test_ds: Dataset = DATASETS[self.cfg.dataset](train=False)

    @property
    def name(self) -> str:
        cfg = self.cfg
        return f"{cfg.model} on {cfg.dataset}, {cfg.loss}, {cfg.optimizer} lr {cfg.lr:g}, steps {cfg.start}-{cfg.n_steps}"

    def build(self, lr: float | None = None) -> tuple[torch.nn.Module, OptimState]:
        """The model and its initial state (with lr `lr`, default cfg.lr)."""
        cfg = self.cfg
        lr = cfg.lr if lr is None else lr
        if lr is None:
            raise click.UsageError(f"no learning rate in {self.dir / 'config.json'}: run experiments/tune_lr.py first")
        torch.manual_seed(cfg.init_seed)
        model = MODELS[cfg.model](self.train_ds.in_shape, self.train_ds.out_dim).to(self.device)
        if cfg.dtype == "bf16":  # the model's forward runs in bfloat16, its outputs (so the loss) in float32
            forward, device_type = model.forward, torch.device(self.device).type

            def forward_bf16(x):
                with torch.autocast(device_type, torch.bfloat16):
                    return forward(x).float()

            model.forward = forward_bf16
        params = {k: p.detach().clone() for k, p in model.named_parameters()}
        if cfg.optimizer == "adam":
            betas = (0.9 if cfg.momentum is None else cfg.momentum, 0.999)
            state = AdamState(params=params, lr=lr, lr_mults=adam_lr_mults(model), betas=betas, weight_decay=cfg.weight_decay)
        elif cfg.momentum is not None:
            state = SGDMomentumState(params=params, lr=lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        else:
            state = SGDState(params=params, lr=lr, weight_decay=cfg.weight_decay)
        return model, state

    def magic(self, model: torch.nn.Module, weights: torch.Tensor | None = None, logger: Logger | None = None, **kwargs) -> Magic:
        """Training (and MAGIC) of `model` with the per-example train weights `weights` (None: all ones) from step cfg.start on."""
        loader = Loader(self.train_ds, self.cfg.batch_size, seed=self.cfg.data_seed, device=self.device,
                        weights=weights, weights_from=self.cfg.start)
        return Magic(model, loader, self.loss_fn, logger, **kwargs)

    def eval_loader(self, subset: Subset) -> EvalLoader:
        """The examples of `subset` of the test set."""
        return EvalLoader(self.test_ds, subset.mask(self.test_ds).nonzero()[:, 0], device=self.device)


def parse_labels(ctx, param, value: str) -> list[int | None]:
    return [None if v.strip().lower() in ("none", "all") else int(v) for v in value.split(",")]


exp_dir = click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
device_option = click.option("--device", type=str, default="cpu", show_default=True)
show_option = click.option("--show/--no-show", default=True, help="show the plots")
labels_option = click.option("--labels", type=str, default="all", callback=parse_labels, help="comma-separated labels, 'all' = any class")


@click.group(help=__doc__)
def main():
    pass


@main.command(help="Create the experiment DIR and save its settings to DIR/config.json.")
@exp_dir
@click.option("--dataset", type=click.Choice(list(DATASETS)), default="mnist", show_default=True)
@click.option("--model", type=click.Choice(list(MODELS)), default="cnn", show_default=True)
@click.option("--loss", type=click.Choice(list(LOSSES)), default="ce", show_default=True)
@click.option("--init-seed", type=int, default=0, show_default=True)
@click.option("--data-seed", type=int, default=0, show_default=True, help="seed of the batch order")
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--optimizer", type=click.Choice(["sgd", "adam"]), default="sgd", show_default=True)
@click.option("--lr", type=float, default=None, help="learning rate (default: set by experiments/tune_lr.py)")
@click.option("--momentum", type=float, default=None, help="SGD momentum (default: none) or Adam's beta1 (default: 0.9)")
@click.option("--weight-decay", type=float, default=0.0, show_default=True)
@click.option("--n-steps", type=int, default=400, show_default=True)
@click.option("--start", type=int, default=0, show_default=True, help="perturb / differentiate the training weights from this step on (trained on all the data before)")
@click.option("--dtype", type=click.Choice(["float32", "float64", "bf16"]), default="float32", show_default=True, help="bf16: mixed precision (float32 parameters, bfloat16 forward)")
def new(exp_dir: Path, **kwargs):
    cfg = ExperimentConfig(**kwargs)
    path = exp_dir / "config.json"
    if path.exists():
        raise click.UsageError(f"{path} exists")
    exp_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2))
    click.echo(f"saved {path}" + ("" if cfg.lr else " (no lr: run experiments/tune_lr.py)"))


@main.command(help="Train the full model (logged, into DIR/full_model; skipped if it exists) and, into DIR/retrained, one model "
              "per (label, index) with the training weights perturbed from step `start` on: dropping a random --alpha fraction "
              "of the data (--kind drop), or adding N(0, alpha) to every weight (--kind randn; same ||w - 1|| as drop). Existing models are skipped.")
@exp_dir
@device_option
@click.option("--kind", type=click.Choice(["drop", "randn"]), default="drop", show_default=True)
@click.option("--alpha", type=float, default=0.05, show_default=True, help="size of the perturbation")
@labels_option
@click.option("--n-subsets", type=int, default=10, show_default=True, help="perturbations per label (indices 0, ..., n-1)")
@click.option("--num-top-eigs", type=int, default=1, show_default=True, help="full model only: log the top Hessian eigenvalues (0: don't)")
@click.option("--log-tangent/--no-log-tangent", default=True, help="full model only: log the norms of ds_t/ds_0 v for a random v (forward-mode AD)")
@click.option("--eval-every", type=int, default=50, show_default=True, help="full model only: log the test loss and metrics every this many steps (0: never)")
@click.option("--eval-frac", type=float, default=0.1, show_default=True, help="full model only: fraction of the test set (a fixed random subset) the test loss is logged on")
@show_option
def retrain(exp_dir, device, kind, alpha, labels, n_subsets, num_top_eigs, log_tangent, eval_every, eval_frac, show):
    """Only the full model is logged; the retrained models only save their final state."""
    exp = Experiment(exp_dir, device)
    cfg = exp.cfg
    model, init_state = exp.build()

    full_dir = exp_dir / "full_model"
    if not (full_dir / "final.pt").exists():
        click.echo("training the full model")
        full_dir.mkdir(parents=True, exist_ok=True)
        logger = Logger(exp.name)
        magic = exp.magic(
            model, logger=logger, test_loader=exp.eval_loader(Subset(frac=eval_frac)) if eval_every else None,
            eval_every=eval_every, num_top_eigs=num_top_eigs, log_tangent=log_tangent,
        )
        final, _ = magic.forward_pass(init_state, n_steps=cfg.n_steps, log=True)
        torch.save(asdict(final), full_dir / "final.pt")
        logger.finish(full_dir / "metrics.pt")
        if show:
            plt.show()
        plt.close("all")

    labels = [None] if kind == "randn" else labels
    perturbations = [Perturbation(kind=kind, alpha=alpha, label=label, index=i) for label in labels for i in range(n_subsets)]
    for p in tqdm(perturbations, desc="retrained models"):
        run_dir = exp_dir / "retrained" / p.dirname
        if (run_dir / "final.pt").exists():
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        final, _ = exp.magic(model, weights=p.weights(exp.train_ds)).forward_pass(init_state, n_steps=cfg.n_steps)
        torch.save(asdict(final), run_dir / "final.pt")
        (run_dir / "perturbation.json").write_text(json.dumps(asdict(p)))


@main.command("run-magic", help="Train on all the data and compute df/dw (MAGIC) for the loss f on each eval subset after training, "
              "for the training weights of the steps from `start` on. Saves to DIR/magic; existing eval subsets are skipped.")
@exp_dir
@device_option
@labels_option
@click.option("--frac", type=float, default=0.1, show_default=True, help="fraction of the test set in each eval subset (all from one label for label subsets)")
@click.option("--n-subsets", type=int, default=1, show_default=True, help="eval subsets per label (indices 0, ..., n-1)")
@click.option("--passes", type=int, default=2, show_default=True, help="re-trainings to reconstruct the states in reverse (memory: passes * steps^(1/passes) states)")
@show_option
def run_magic(exp_dir, device, labels, frac, n_subsets, passes, show):
    exp = Experiment(exp_dir, device)
    cfg = exp.cfg
    model, init_state = exp.build()
    magic_dir = exp_dir / "magic"
    subsets = [Subset(frac=frac, label=label, index=i) for label in labels for i in range(n_subsets)]
    subsets = [s for s in subsets if not (magic_dir / s.dirname / "linearization.pt").exists()]
    click.echo(f"{len(subsets)} new eval subsets")
    if not subsets:
        return
    logger = Logger(f"MAGIC: norms of df/ds_t, f = loss on {frac:.1%} of the test set; {exp.name}")
    magic = exp.magic(model, logger=logger)
    if cfg.start > 0:  # the state at `start`
        init_state, _ = magic.forward_pass(init_state, n_steps=cfg.start)
    outputs, betas = magic.backward_pass(
        init_state, n_steps=cfg.n_steps - cfg.start, start=cfg.start,
        eval_loaders={s.name: exp.eval_loader(s) for s in subsets}, passes=passes,
    )
    magic_dir.mkdir(parents=True, exist_ok=True)
    logger.finish(magic_dir / "metrics.pt")
    for s in subsets:
        subset_dir = magic_dir / s.dirname
        subset_dir.mkdir(exist_ok=True)
        torch.save({"output": outputs[s.name].cpu(), "beta": betas[s.name].cpu()}, subset_dir / "linearization.pt")  # f, df/dw [N]
        (subset_dir / "subset.json").write_text(json.dumps(asdict(s)))
        click.echo(f"  {s.name}: f = {outputs[s.name].item():.4f}")
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
