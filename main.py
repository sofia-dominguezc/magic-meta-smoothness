"""

  python main.py new DIR [settings]   # create an experiment: save its settings (dataset, model, optimizer, start, ...) to DIR/config.json
  python main.py train DIR [...]      # train the full model with logs (loss, norms, top Hessian eigenvalues, tangent, ...)
  python main.py retrain DIR [...]    # train models with perturbed training weights (data dropped / noised), vmapped together
  python main.py run-magic DIR [...]  # train on all the data and compute df/dw for each eval set


Everything saved in DIR is comparable (same config, so same training and same steps `start`-`n_steps` perturbed).
A saved state (*.pt below) is `asdict(OptimState)`, loaded by `Experiment.load_state`:
  DIR/full_model/   start.pt (if start > 0): the state at step `start`, trained once on all the data, from which everything
                    continues; metrics.pt + plots: the logs of `train`; final.pt: the state after training on all the data,
                    trained by `retrain` vmapped with the retrained models (their reference in evaluate.py: the plain,
                    logged and vmapped passes differ by fp32 rounding, see experiments/vmap_error.py)
  DIR/retrained/    one dir per retrained model: final.pt, perturbation.pt = {weights [N], kind, alpha}
  DIR/magic/        one dir per eval set: linearization.pt = {indices [n] (test examples), output f, beta df/dw [N]};
                    metrics.pt + plots (the logged df/ds_t norms) of the last run-magic call
The dir names (<kind>_<alpha>_<index>, <n>_examples_<index>) are labels: nothing is parsed from them.

retrain and run-magic can be run several times (e.g. with other alphas, more eval sets): existing ones are skipped.
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
from src.magic import LOSSES, Magic, TrainLogger
from src.models import MODELS, AdamState, LRSchedule, OptimState, SGDMomentumState, SGDState
from src.logging import Logger


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    dataset: str = "mnist"
    model: str = "cnn"
    loss: str = "ce"
    init_seed: int = 0
    data_seed: int = 0
    n_train: int | None = None  # train on a random subset of this many examples (seeded by data_seed; None: all)
    batch_size: int = 32
    optimizer: str = "sgd"  # sgd, momentum or adam
    lr: float | None = None
    lr_schedule: str = "constant"  # or cosine, after `warmup` linear steps (models.LRSchedule)
    warmup: int = 0
    beta1: float = 0.9  # momentum/adam only
    beta2: float = 0.999  # adam only
    weight_decay: float = 0.0
    n_steps: int = 400
    start: int = 0  # the training weights are perturbed / differentiated from this step on
    dtype: str = "fp32"  # of the states, losses and df/dw: fp32 or fp64; bf16: fp32 states, model forward in bfloat16


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
        return f"{cfg.model} on {data}, {cfg.loss}, {cfg.optimizer} lr {cfg.lr:g}{schedule}, steps {cfg.start}-{cfg.n_steps}"

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
        return MODELS[self.cfg.model](self.train_ds.in_shape, self.train_ds.out_dim).to(self.device)

    @cached_property
    def init_state(self) -> OptimState:
        """The initial state of `model` with the config's optimizer and lr schedule."""
        cfg = self.cfg
        if cfg.lr is None:
            raise click.UsageError(f"no learning rate in {self.dir / 'config.json'}: run experiments/tune_lr.py first")
        schedule = LRSchedule(cfg.lr, kind=cfg.lr_schedule, warmup=cfg.warmup, n_steps=cfg.n_steps)
        params = {k: p.detach().clone() for k, p in self.model.named_parameters()}
        if cfg.optimizer == "adam":
            return AdamState(params=params, lr=schedule, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
        if cfg.optimizer == "momentum":
            return SGDMomentumState(params=params, lr=schedule, weight_decay=cfg.weight_decay, momentum=cfg.beta1)
        assert cfg.optimizer == "sgd", cfg.optimizer
        return SGDState(params=params, lr=schedule, weight_decay=cfg.weight_decay)

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
        state, _ = self.magic.train(self.init_state, n_steps=self.cfg.start)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(asdict(state), path)
        return state

    def load_state(self, path: Path) -> OptimState:
        """A state saved as `asdict(state)` (on CPU)."""
        return type(self.init_state)(**torch.load(path, weights_only=True))


exp_dir = click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
device_option = click.option("--device", type=str, default="cpu", show_default=True)
compile_option = click.option("--compile/--no-compile", default=False, help="torch.compile the plain (unlogged) training passes")
show_option = click.option("--show/--no-show", default=True, help="show the plots")


@click.group(help=__doc__)
def main():
    pass


@main.command(help="Create the experiment DIR and save its settings to DIR/config.json.")
@exp_dir
@click.option("--dataset", type=click.Choice(list(DATASETS)), default="mnist", show_default=True)
@click.option("--model", type=click.Choice(list(MODELS)), default="cnn", show_default=True)
@click.option("--loss", type=click.Choice(list(LOSSES)), default="ce", show_default=True)
@click.option("--init-seed", type=int, default=0, show_default=True)
@click.option("--data-seed", type=int, default=0, show_default=True, help="seed of the batch order (and of the --n-train subset)")
@click.option("--n-train", type=int, default=None, help="train on a random subset of this many examples (default: all)")
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--optimizer", type=click.Choice(["sgd", "momentum", "adam"]), default="sgd", show_default=True)
@click.option("--lr", type=float, default=None, help="learning rate (default: set by experiments/tune_lr.py)")
@click.option("--lr-schedule", type=click.Choice(LR_SCHEDULES), default="constant", show_default=True, help=f"cosine: from lr to {LRSchedule.FINAL_FRACTION:g} lr at n_steps")
@click.option("--warmup", type=int, default=0, show_default=True, help="linear warmup of the lr over this many steps")
@click.option("--beta1", type=float, default=0.9, show_default=True, help="the momentum of momentum, beta1 of adam")
@click.option("--beta2", type=float, default=0.999, show_default=True, help="beta2 of adam")
@click.option("--weight-decay", type=float, default=0.0, show_default=True)
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


@main.command(help="Train the full model on all the data from step `start` with logs, saved to DIR/full_model (metrics.pt and plots).")
@exp_dir
@device_option
@compile_option
@click.option("--num-top-eigs", type=int, default=1, show_default=True, help="log the top Hessian eigenvalues (0: don't)")
@click.option("--num-neg-eigs", type=int, default=0, show_default=True, help="log the most negative Hessian eigenvalues (0: don't)")
@click.option("--eig-every", type=int, default=10, show_default=True, help="log the eigenvalues every this many steps")
@click.option("--log-tangent/--no-log-tangent", default=True, help="log the norms of ds_t/ds_0 v for a random v (forward-mode AD)")
@click.option("--eval-every", type=int, default=100, show_default=True, help="log the test loss and metrics every this many steps (0: never)")
@click.option("--n-test", type=int, default=1000, show_default=True, help="test examples (a fresh random subset at each eval, the whole test set at the last step) the test loss is logged on (0: always the whole test set)")
@show_option
def train(exp_dir, device, compile, num_top_eigs, num_neg_eigs, eig_every, log_tangent, eval_every, n_test, show):
    exp = Experiment(exp_dir, device, compile)
    cfg = exp.cfg
    log = TrainLogger(Logger(exp.name), exp.test_loader, n_top_eigs=num_top_eigs, n_neg_eigs=num_neg_eigs, eig_every=eig_every,
                   tangent=log_tangent, eval_every=eval_every, n_test=n_test or None)
    full_dir = exp_dir / "full_model"
    full_dir.mkdir(parents=True, exist_ok=True)
    exp.magic.train_logged(exp.start_state, n_steps=cfg.n_steps - cfg.start, start=cfg.start, log=log)
    log.logger.finish(full_dir / "metrics.pt")
    if show:
        plt.show()
    plt.close("all")


@main.command(help="Train, into DIR/retrained, --n-subsets models with the training weights perturbed from step `start` on: "
              "dropping a random --alpha fraction of the data (--kind drop), or adding N(0, alpha) to every weight (--kind randn; "
              "same ||w - 1|| as drop), vmapped together with, the first time, the unperturbed model (DIR/full_model/final.pt). "
              "Existing models are skipped.")
@exp_dir
@device_option
@compile_option
@click.option("--kind", type=click.Choice(["drop", "randn"]), default="drop", show_default=True)
@click.option("--alpha", type=float, default=0.05, show_default=True, help="size of the perturbation")
@click.option("--n-subsets", type=int, default=10, show_default=True, help="perturbations (indices 0, ..., n-1)")
@click.option("--vmap-limit", type=int, default=None, help="train at most this many models per vmapped pass (default: all at once)")
def retrain(exp_dir, device, compile, kind, alpha, n_subsets, vmap_limit):
    exp = Experiment(exp_dir, device, compile)
    cfg = exp.cfg
    runs = {i: exp_dir / "retrained" / f"{kind}_{alpha:g}_{i}" for i in range(n_subsets)}  # index -> dir
    runs = {i: d for i, d in runs.items() if not (d / "final.pt").exists()}
    weights = [exp.train_ds.weights(kind, alpha, index=i) for i in runs]
    paths = [d / "final.pt" for d in runs.values()]
    if not (full_path := exp_dir / "full_model" / "final.pt").exists():  # the unperturbed reference, trained the same way
        weights.append(torch.ones(len(exp.train_ds)))
        paths.append(full_path)
    if not weights:
        return
    limit = vmap_limit or len(weights)
    click.echo(f"training {len(weights)} models" + (f" in chunks of {limit}" if limit < len(weights) else ""))
    for c in range(0, len(weights), limit):
        final, _ = exp.magic.train(exp.start_state, n_steps=cfg.n_steps - cfg.start, start=cfg.start, weights=torch.stack(weights[c:c + limit]))
        for r, path in enumerate(paths[c:c + limit]):
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(asdict(final.apply(lambda t: t[r])), path)
    for d, w in zip(runs.values(), weights):
        torch.save({"weights": w, "kind": kind, "alpha": alpha}, d / "perturbation.pt")


@main.command("run-magic", help="Train on all the data and compute df/dw (MAGIC) for the loss f on each eval set after training, "
              "for the training weights of the steps from `start` on. Saves to DIR/magic; existing eval sets are skipped.")
@exp_dir
@device_option
@compile_option
@click.option("--n-test", type=int, default=10, show_default=True, help="test examples in each eval set (a fixed random subset): f is their mean loss")
@click.option("--n-subsets", type=int, default=1, show_default=True, help="eval sets (indices 0, ..., n-1)")
@click.option("--passes", type=int, default=2, show_default=True, help="re-trainings to reconstruct the states in reverse (memory: passes * steps^(1/passes) states)")
@show_option
def run_magic(exp_dir, device, compile, n_test, n_subsets, passes, show):
    exp = Experiment(exp_dir, device, compile)
    cfg = exp.cfg
    magic_dir = exp_dir / "magic"
    sets = {i: (f"{n_test} examples #{i}", magic_dir / f"{n_test}_examples_{i}") for i in range(n_subsets)}  # index -> (name, dir)
    sets = {i: (name, d) for i, (name, d) in sets.items() if not (d / "linearization.pt").exists()}
    eval_sets = {name: exp.test_ds.subset(n_test, index=i) for i, (name, _) in sets.items()}  # name -> test examples
    click.echo(f"backpropagating {len(eval_sets)} eval sets")
    if not eval_sets:
        return
    logger = Logger(f"MAGIC: norms of df/ds_t, f = loss on {n_test} test examples; {exp.name}")
    outputs, betas = exp.magic.backprop(
        exp.start_state, n_steps=cfg.n_steps - cfg.start, start=cfg.start, test_loader=exp.test_loader, eval_sets=eval_sets, passes=passes, logger=logger,
    )
    magic_dir.mkdir(parents=True, exist_ok=True)
    logger.finish(magic_dir / "metrics.pt")
    for name, d in sets.values():
        d.mkdir(exist_ok=True)
        torch.save({"indices": eval_sets[name], "output": outputs[name].cpu(), "beta": betas[name].cpu()}, d / "linearization.pt")
        click.echo(f"  {name}: f = {outputs[name].item():.4f}")
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
