"""

  python main.py new DIR [settings]   # create an experiment: save its settings (dataset, model, optimizer, start, ...) to DIR/config.json
  python main.py train DIR [...]      # train the full model (with the default logs), then models with perturbed training weights (data dropped / noised), vmapped together
  python main.py magic DIR [...]      # train on all the data and compute df/dw for each eval set
  python train_logged.py DIR [...]    # train the full model once more, only for the detailed logs (Hessian eigenvalues, tangent, EMA, test loss)


Everything saved in DIR is comparable (same config, so same training and same steps `start`-`n_steps` perturbed).
A saved state (*.pt below) is `asdict(OptimState)`, loaded by `Experiment.load_state`:
  DIR/full_model/   start.pt (if start > 0): the state at step `start`, trained once on all the data, from which everything
                    continues; final.pt: the state after training on all the data (the retrained models' reference in
                    evaluate.py: the plain and vmapped passes differ by fp32 rounding, see experiments/vmap_error.py);
                    metrics.pt + plots: the default logs of that training (loss, metrics, op norms of the params and grads)
  DIR/retrained/    one dir per retrained model: final.pt, perturbation.pt = {weights [N], kind, alpha}
  DIR/magic/        one dir per eval set: linearization.pt = {indices [n] (test examples), output f, beta df/dw [N]};
                    metrics.pt + plots: the norms of df/ds_t and df/dw_t of the last `magic` call
  DIR/logged/       metrics.pt + plots of train_logged.py
The dir names (<kind>_<alpha>_<index>, <n>_examples_<index>) are labels: nothing is parsed from them.

train and magic can be run several times (e.g. with other alphas, more eval sets): existing models / eval sets are skipped.
experiments/ compares them (experiments/evaluate.py), tunes the lr (experiments/tune_lr.py; needed if `new` is given no
--lr) and runs whole sweeps (experiments/sweep.py).
"""

import json
import click
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import asdict, dataclass, field
from functools import cached_property

import torch

from src.data import DATASETS, Dataset, Loader
from src.magic import Magic
from src.utils import LOSSES
from src.models import MODELS
from src.optimizers import AdamState, LRSchedule, NewtonState, OptimState, SGDMomentumState, SGDState, SharpnessState
from src.logging import Logger


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    dataset: str = "mnist"
    model: str = "cnn"
    width: int | None = None  # of the model (mlp: hidden, cnn: channels of the 1st stage, vit: embedding; None: the model's default)
    loss: str = "ce"
    init_seed: int = 0
    data_seed: int = 0
    n_train: int | None = None  # train on a random subset of this many examples (seeded by data_seed; None: all)
    batch_size: int = 32
    optimizer: str = "sgd"  # sgd, momentum, adam, sharpness (SGD with the step size lr / top Hessian eigenvalue) or newton
    damping: float = 0.01  # newton only
    lr: float | None = None
    lr_schedule: str = "constant"  # or cosine, after `warmup` linear steps (models.LRSchedule)
    warmup: int = 0
    beta1: float = 0.9  # momentum/adam only
    beta2: float = 0.999  # adam only
    weight_decay: float = 0.0
    n_steps: int = 400
    start: int = 0  # train for this many steps before changing weights/logging
    dtype: str = "fp32"  # fp32 or fp64. bf16: fp32 states with bfloat16 autocast


DTYPES = {"fp32": torch.float32, "fp64": torch.float64, "bf16": torch.float32}  # of the states
LR_SCHEDULES = ["constant", "cosine"]


@dataclass
class Experiment:
    """An experiment dir with its saved config (`cfg`, read from DIR/config.json unless given), and everything built
    from it, each built once when first used: the datasets and loaders, the model and its initial state (deterministic),
    `magic`, and `start_state` (the state at step cfg.start). `compile` torch.compiles the plain training passes."""

    dir: Path
    device: str = "cpu"
    compile: bool = False
    cfg: ExperimentConfig = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if self.cfg is None:
            self.cfg = ExperimentConfig(**json.loads((self.dir / "config.json").read_text()))
        torch.set_default_dtype(DTYPES[self.cfg.dtype])

    @property
    def name(self) -> str:
        cfg = self.cfg
        schedule = ("" if cfg.lr_schedule == "constant" else f" {cfg.lr_schedule}") + (f" warmup {cfg.warmup}" if cfg.warmup else "")
        data = cfg.dataset if cfg.n_train is None else f"{cfg.n_train} examples of {cfg.dataset}"
        model = cfg.model if cfg.width is None else f"{cfg.model} (width {cfg.width})"
        return f"{model} on {data}, {cfg.loss}, {cfg.optimizer} lr {cfg.lr:g}{schedule}, steps {cfg.start}-{cfg.n_steps}"

    @property
    def bf16(self) -> bool:
        return self.cfg.dtype == "bf16"

    @property
    def loss_fn(self):
        return LOSSES[self.cfg.loss]

    @cached_property
    def train_ds(self) -> Dataset:
        ds = DATASETS[self.cfg.dataset](train=True)
        if self.cfg.n_train is not None:
            ds.take(self.cfg.n_train, seed=self.cfg.data_seed)
        return ds

    @cached_property
    def test_ds(self) -> Dataset:
        return DATASETS[self.cfg.dataset](train=False)

    @cached_property
    def loader(self) -> Loader:
        return Loader(self.train_ds, self.cfg.batch_size, seed=self.cfg.data_seed, device=self.device)

    @cached_property
    def test_loader(self) -> Loader:
        return Loader(self.test_ds, batch_size=256, device=self.device)

    @cached_property
    def model(self) -> torch.nn.Module:
        torch.manual_seed(self.cfg.init_seed)
        width = {} if self.cfg.width is None else {"width": self.cfg.width}
        return MODELS[self.cfg.model](self.train_ds.in_shape, self.train_ds.out_dim, **width).to(self.device)

    @cached_property
    def init_state(self) -> OptimState:
        """The initial state of `model` with the config's optimizer and lr schedule."""
        cfg = self.cfg
        if cfg.lr is None:
            raise click.UsageError(f"no learning rate in {self.dir / 'config.json'}: run experiments/tune_lr.py first")
        schedule = LRSchedule(cfg.lr, kind=cfg.lr_schedule, warmup=cfg.warmup, n_steps=cfg.n_steps)
        params = {k: p.detach().clone() for k, p in self.model.named_parameters()}
        if cfg.optimizer == "adam":
            return AdamState(params=params, lr=schedule, betas=(cfg.beta1, cfg.beta2), wd=cfg.weight_decay)
        if cfg.optimizer == "momentum":
            return SGDMomentumState(params=params, lr=schedule, wd=cfg.weight_decay, momentum=cfg.beta1)
        if cfg.optimizer == "sharpness":
            return SharpnessState(params=params, lr=schedule, wd=cfg.weight_decay)
        if cfg.optimizer == "newton":
            return NewtonState(params=params, lr=schedule, wd=cfg.weight_decay, damping=cfg.damping)
        assert cfg.optimizer == "sgd", cfg.optimizer
        return SGDState(params=params, lr=schedule, wd=cfg.weight_decay)

    @cached_property
    def magic(self) -> Magic:
        return Magic(model=self.model, loader=self.loader, loss_fn=self.loss_fn, bf16=self.bf16, compile=self.compile)

    @cached_property
    def start_state(self) -> OptimState:
        """The state at step cfg.start: `init_state` trained on all the data (saved in DIR/full_model/start.pt)."""
        if self.cfg.start == 0:
            return self.init_state
        path = self.dir / "full_model" / "start.pt"
        if path.exists():
            return self.load_state(path)
        click.echo(f"training the base model to step {self.cfg.start}")
        state = self.magic.train(self.init_state, n_steps=self.cfg.start)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(asdict(state), path)
        return state

    def load_state(self, path: Path) -> OptimState:
        """A state saved as `asdict(state)` (on CPU)."""
        return type(self.init_state)(**torch.load(path, weights_only=True))


exp_dir = click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
device_option = click.option("--device", type=str, default="cpu", show_default=True)
compile_option = click.option("--compile/--no-compile", default=False, help="torch.compile the model forward of the training passes")
show_option = click.option("--show/--no-show", default=True, help="show the plots")


@click.group(help=__doc__)
def main():
    pass


@main.command(help="Create the experiment DIR and save its settings to DIR/config.json.")
@exp_dir
@click.option("--dataset", type=click.Choice(list(DATASETS)), default="mnist", show_default=True)
@click.option("--model", type=click.Choice(list(MODELS)), default="cnn", show_default=True)
@click.option("--loss", type=click.Choice(list(LOSSES)), default="ce", show_default=True)
@click.option("--width", type=int, default=None, help="of the model (mlp: hidden, cnn: channels of the 1st stage, vit: embedding; default: 512 / 32 / 96; cnn 6 ~ 11k params)")
@click.option("--init-seed", type=int, default=0, show_default=True)
@click.option("--data-seed", type=int, default=0, show_default=True, help="seed of the batch order (and of the --n-train subset)")
@click.option("--n-train", type=int, default=None, help="train on a random subset of this many examples (default: all)")
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--optimizer", type=click.Choice(["sgd", "momentum", "adam", "sharpness", "newton"]), default="sgd", show_default=True, help="sharpness: SGD with the step size lr * 2 / sharpness")
@click.option("--lr", type=float, default=None, help="learning rate (default: set by experiments/tune_lr.py)")
@click.option("--lr-schedule", type=click.Choice(LR_SCHEDULES), default="constant", show_default=True, help=f"cosine: from lr to {LRSchedule.FINAL_FRACTION:g} lr at n_steps")
@click.option("--warmup", type=int, default=0, show_default=True, help="linear warmup of the lr over this many steps")
@click.option("--beta1", type=float, default=0.9, show_default=True, help="the momentum of momentum, beta1 of adam")
@click.option("--beta2", type=float, default=0.999, show_default=True, help="beta2 of adam")
@click.option("--weight-decay", type=float, default=0.0, show_default=True)
@click.option("--damping", type=float, default=0.1, show_default=True, help="newton: added to the Hessian's diagonal")
@click.option("--n-steps", type=int, default=400, show_default=True)
@click.option("--start", type=int, default=0, show_default=True, help="perturb / differentiate the training weights from this step on (trained on all the data before)")
@click.option("--dtype", type=click.Choice(list(DTYPES)), default="fp32", show_default=True, help="of the states, losses and df/dw; bf16 = fp32 states, model forward in bfloat16")
def new(exp_dir: Path, **kwargs):
    cfg = ExperimentConfig(**kwargs)
    path = exp_dir / "config.json"
    if path.exists():
        raise click.UsageError(f"{path} exists")
    exp_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2))
    click.echo(f"saved {path}" + ("" if cfg.lr else " (no lr: run experiments/tune_lr.py)"))


@main.command(help="Train the full model on all the data from step `start` (DIR/full_model: final.pt, metrics.pt and plots of the loss, "
              "metrics and op norms of the params and grads; skipped if final.pt exists), then --n-subsets models with the training weights "
              "perturbed from step `start` on, into DIR/retrained: dropping a random --alpha fraction of the data (--kind drop), or adding "
              "N(0, alpha) to every weight (--kind randn; same ||w - 1|| as drop), vmapped together. Existing models are skipped.")
@exp_dir
@device_option
@compile_option
@click.option("--kind", type=click.Choice(["drop", "randn"]), default="randn", show_default=True)
@click.option("--alpha", "alphas", type=float, multiple=True, default=[0.05], show_default=True, help="size of the perturbation (repeatable: all the models in one vmapped pass)")
@click.option("--n-subsets", type=int, default=0, show_default=True, help="perturbed models (indices 0, ..., n-1; 0: only the full model)")
@click.option("--vmap-limit", type=int, default=None, help="train at most this many models per vmapped pass (default: all at once)")
@show_option
def train(exp_dir, device, compile, kind, alphas, n_subsets, vmap_limit, show):
    exp = Experiment(exp_dir, device, compile)
    cfg = exp.cfg
    n_steps = cfg.n_steps - cfg.start
    full_dir = exp_dir / "full_model"
    if not (full_dir / "final.pt").exists():
        logger = Logger(exp.name)
        final = exp.magic.train(exp.start_state, n_steps=n_steps, start=cfg.start, log=logger)
        full_dir.mkdir(parents=True, exist_ok=True)
        torch.save(asdict(final), full_dir / "final.pt")
        logger.finish(full_dir / "metrics.pt")

    runs = {(alpha, i): exp_dir / "retrained" / f"{kind}_{alpha:g}_{i}" for alpha in alphas for i in range(n_subsets)}  # -> dir
    runs = {key: d for key, d in runs.items() if not (d / "final.pt").exists()}
    weights = [exp.train_ds.weights(kind, alpha, index=i) for alpha, i in runs]
    paths = [d / "final.pt" for d in runs.values()]
    limit = vmap_limit or len(weights) or 1
    if weights:
        click.echo(f"training {len(weights)} perturbed models" + (f" in chunks of {limit}" if limit < len(weights) else ""))
    for c in range(0, len(weights), limit):
        final = exp.magic.train(exp.start_state, n_steps=n_steps, start=cfg.start, weights=torch.stack(weights[c:c + limit]))
        for r, path in enumerate(paths[c:c + limit]):
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(asdict(final.apply(lambda t: t[r])), path)
    for ((alpha, _), d), w in zip(runs.items(), weights):
        torch.save({"weights": w, "kind": kind, "alpha": alpha}, d / "perturbation.pt")
    if show:
        plt.show()
    plt.close("all")


@main.command(help="Train on all the data and compute df/dw (MAGIC) for the loss f on each eval set after training, "
              "for the training weights of the steps from `start` on. Saves to DIR/magic (linearization.pt per eval set; metrics.pt and "
              "plots of the norms of df/ds_t and df/dw_t); existing eval sets are skipped.")
@exp_dir
@device_option
@compile_option
@click.option("--n-test", type=int, default=10, show_default=True, help="test examples in each eval set (a fixed random subset): f is their mean loss")
@click.option("--n-subsets", type=int, default=1, show_default=True, help="eval sets (indices 0, ..., n-1)")
@click.option("--passes", type=int, default=2, show_default=True, help="re-trainings to reconstruct the states in reverse (memory: passes * steps^(1/passes) states)")
@click.option("--vmap-limit", type=int, default=None, help="backpropagate at most this many eval sets at a time, a replay each (default: all at once; only the first chunk is logged)")
@show_option
def magic(exp_dir, device, compile, n_test, n_subsets, passes, vmap_limit, show):
    exp = Experiment(exp_dir, device, compile)
    cfg = exp.cfg
    magic_dir = exp_dir / "magic"
    dirs = {i: magic_dir / f"{n_test}_examples_{i}" for i in range(n_subsets)}  # index -> dir
    dirs = {i: d for i, d in dirs.items() if not (d / "linearization.pt").exists()}
    if not dirs:
        return
    indices = [exp.test_ds.subset(n_test, index=i) for i in dirs]  # the test examples of each eval set
    limit = vmap_limit or len(dirs)
    click.echo(f"backpropagating {len(dirs)} eval sets" + (f" in chunks of {limit}" if limit < len(dirs) else ""))
    logger = Logger(f"MAGIC: df/ds_t and df/dw_t; {exp.name}", smooth=False)
    eval_sets = [exp.test_ds.examples(idx) for idx in indices]
    outputs, betas = [], []  # per chunk: f(s_T) [#fns], df/dw [#fns, N]
    for c in range(0, len(eval_sets), limit):
        output, beta = exp.magic.backprop(
            exp.start_state, n_steps=cfg.n_steps - cfg.start, start=cfg.start, eval_sets=eval_sets[c:c + limit],
            passes=passes, log=logger if c == 0 else None,
        )
        outputs.append(output)
        betas.append(beta)
    for (i, d), idx, output, beta in zip(dirs.items(), indices, torch.cat(outputs), torch.cat(betas)):
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"indices": idx, "output": output.cpu(), "beta": beta.cpu()}, d / "linearization.pt")
        click.echo(f"  {n_test} examples #{i}: f = {output.item():.4f}")
    logger.finish(magic_dir / "metrics.pt")
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
