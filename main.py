"""

  python main.py new DIR [settings]   # create an experiment: save its settings (dataset, model, optimizer, start, ...) to DIR/config.json
  python main.py train DIR [...]      # train the full model on all the data, with the logs (loss, metrics, norms, EMA; optionally Hessian eigenvalues, curvature along the gradient / tangent / EMA direction, tangent, test loss)
  python main.py retrain DIR [...]    # train models with perturbed training weights (data dropped / noised), vmapped together
  python main.py magic DIR [...]      # compute df/dw of the trained full model for each eval set
  python main.py prune DIR [...]      # delete the saved states (the retrained and full models' *.pt) to free disk space, keeping everything else (DIR: an experiment or a dir of them)
train and retrain are independent (each only needs DIR/config.json); magic needs train; experiments/evaluate.py needs all of them.

Everything saved in DIR is comparable (same config, so same training and same steps `start`-`n_steps` perturbed).
A saved state (*.pt below) is `asdict(OptimState)`, loaded by `Experiment.load_state`:
  DIR/full_model/   start.pt (if start > 0): the state at step `start`, trained once on all the data, from which everything
                    continues; final.pt: the state after training on all the data (the retrained models' reference in
                    evaluate.py);
                    checkpoints/<t>.pt (`train --checkpoint-every`): the state at the optimizer step t, from which `magic`
                    replays the steps up to the next one; resume.pt: the EMA and tangent at the last checkpoint of a killed
                    `train`, which the next `train` continues from;
                    metrics.pt + plots: the logs of `train` (see `magic.TrainLog`), rewritten at every checkpoint
  DIR/retrained/    one dir per retrained model: final.pt, perturbation.pt = {weights [N], kind, alpha};
                    losses.json: its loss on every eval set evaluated so far (`Experiment.eval_losses`, also in DIR/full_model:
                    the evaluations read it instead of the state, so `prune` can delete the states)
  DIR/magic/        one dir per eval set: linearization.pt = {indices [n] (test examples), output f, d_weights df/dw [N]};
                    metrics.pt + plots: the norms of df/ds_t and df/dw_t of the last `magic` call, rewritten at every checkpoint;
                    resume.pt: df/ds_t and df/dw so far at the last checkpoint a killed `magic` reached, which the next
                    `magic` of the same eval sets continues from
The dir names (<kind>_<alpha>_<index>, <n>_examples_<index>) are labels: nothing is parsed from them.

retrain and magic can be run several times (e.g. with other alphas, more eval sets): existing models / eval sets are skipped;
train skips an existing full model unless --force. experiments/ compares them (experiments/evaluate.py), tunes the lr
(experiments/tune_lr.py; needed if `new` is given no --lr) and runs whole sweeps (experiments/sweep.py).
"""

import hashlib
import json
import shutil
import click
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import asdict, dataclass, field
from functools import cached_property

import torch
from torch import Tensor

from src.data import DATASETS, Dataset, Images, Loader
from src.magic import Magic, TrainLog, backprop_logged, to_device
from src.objective import LOSSES, Objective
from src.models import MODELS
from src.optimizers import AdamState, HeunState, LRSchedule, OptimState, SGDMomentumState, SGDState
from src.second_order import CentralFlowState, EdgeFlowState, NewtonState, SecondOrderState, SharpnessState
from src.logging import Logger, fig_title


def parse_devices(device: str) -> list[str]:
    """--device: "cpu", "cuda", or a number of GPUs ("8" -> cuda:0, ..., cuda:7)"""
    return [f"cuda:{i}" for i in range(int(device))] if device.isdigit() else device.split(",")


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    dataset: str = "mnist"
    model: str = "cnn"
    width: int | None = None  # of the model (mlp: hidden, cnn: channels of the 1st stage, vit: embedding; None: the model's default)
    depth: int | None = None  # of the model (mlp: hidden layers, cnn: blocks per stage, vit: blocks; None: the model's default)
    loss: str = "ce"
    init_seed: int = 0
    data_seed: int = 0
    n_train: int | None = None  # train on a random subset of this many examples (seeded by data_seed; None: all)
    n_classes: int | None = None  # keep only the first this many classes of the dataset, train and test (None: all)
    batch_size: int = 32
    optimizer: str = "sgd"  # sgd, momentum, adam, sharpness (SGD with the step size lr / top Hessian eigenvalue), newton, edgeflow, centralflow or heun
    lr: float | None = None
    lr_schedule: str = "constant"  # or cosine, after `warmup` linear steps (models.LRSchedule)
    warmup: int = 0
    beta1: float = 0.9  # momentum/adam only
    beta2: float = 0.999  # adam only
    weight_decay: float = 0.0
    n_steps: int = 400  # gradient descent steps (edge / central flow: SUBSTEPS optimizer steps each, see `Experiment.steps`)
    start: int = 0  # train for this many steps before changing weights/logging
    dtype: str = "fp32"  # fp32 or fp64. bf16: fp32 states with bfloat16 autocast


DTYPES = {"fp32": torch.float32, "fp64": torch.float64, "bf16": torch.float32}  # of the states
LR_SCHEDULES = ["constant", "cosine"]


@dataclass
class Experiment:
    """An experiment dir with its saved config (`cfg`, read from DIR/config.json unless given), and everything built
    from it, each built once when first used: the datasets and loaders, the model and its initial state (deterministic),
    `objective`, `magic`, and `start_state` (the state at step cfg.start). `compile` torch.compiles the plain training passes."""

    dir: Path
    device: str = "cpu"
    compile: bool = False
    cfg: ExperimentConfig = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if self.cfg is None:
            saved = json.loads((self.dir / "config.json").read_text())
            for removed in ("damping", "alpha"):  # (older configs)
                saved.pop(removed, None)
            self.cfg = ExperimentConfig(**saved)
        torch.set_default_dtype(DTYPES[self.cfg.dtype])
        self.device, *self.helper_devices = parse_devices(self.device)

    @property
    def name(self) -> str:
        cfg = self.cfg
        schedule = ("" if cfg.lr_schedule == "constant" else f" {cfg.lr_schedule}") + (f" warmup {cfg.warmup}" if cfg.warmup else "")
        data = cfg.dataset if cfg.n_train is None else f"{cfg.n_train} examples of {cfg.dataset}"
        data += "" if cfg.n_classes is None else f" ({cfg.n_classes} classes)"
        size = [f"{name} {value}" for name, value in (("width", cfg.width), ("depth", cfg.depth)) if value is not None]
        model = cfg.model + (f" ({', '.join(size)})" if size else "")
        return f"{model} on {data}, {cfg.loss}, {cfg.optimizer} lr {cfg.lr:g}{schedule}, steps {cfg.start}-{cfg.n_steps}"

    def dataset(self, train: bool) -> Dataset:
        """The train / test set of the config, limited to its first `n_classes` classes"""
        ds = DATASETS[self.cfg.dataset](train=train)
        if self.cfg.n_classes is not None:
            ds.limit_classes(self.cfg.n_classes)
        return ds

    @cached_property
    def train_ds(self) -> Dataset:
        ds = self.dataset(train=True)
        if self.cfg.n_train is not None:
            ds.take(self.cfg.n_train, seed=self.cfg.data_seed)
        return ds.to(self.device)

    @cached_property
    def test_ds(self) -> Dataset:
        return self.dataset(train=False).to(self.device)

    @property
    def substeps(self) -> int:
        """Optimizer steps per step of the config"""
        return {"edgeflow": EdgeFlowState.SUBSTEPS, "centralflow": CentralFlowState.SUBSTEPS}.get(self.cfg.optimizer, 1)

    @property
    def steps(self) -> tuple[int, int]:
        """(start, n_steps) of the training in optimizer steps"""
        return self.cfg.start * self.substeps, (self.cfg.n_steps - self.cfg.start) * self.substeps

    @cached_property
    def loader(self) -> Loader:
        return Loader(self.train_ds, self.cfg.batch_size, seed=self.cfg.data_seed, device=self.device, repeat=self.substeps)

    @cached_property
    def test_loader(self) -> Loader:
        return Loader(self.test_ds, batch_size=256, device=self.device)

    @cached_property
    def model(self) -> torch.nn.Module:
        torch.manual_seed(self.cfg.init_seed)
        size = {"width": self.cfg.width, "blocks" if self.cfg.model == "cnn" else "depth": self.cfg.depth}
        size = {name: value for name, value in size.items() if value is not None}
        dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)  # the same init in every dtype (the random stream depends on it)
        try:
            model = MODELS[self.cfg.model](self.train_ds.in_shape, self.train_ds.out_dim, **size)
        finally:
            torch.set_default_dtype(dtype)
        return model.to(self.device, dtype)

    @cached_property
    def init_state(self) -> OptimState:
        """The initial state of `model` with the config's optimizer and lr schedule."""
        cfg = self.cfg
        if cfg.lr is None:
            raise click.UsageError(f"no learning rate in {self.dir / 'config.json'}: run experiments/tune_lr.py first")
        schedule = LRSchedule(cfg.lr, kind=cfg.lr_schedule, warmup=cfg.warmup * self.substeps, n_steps=cfg.n_steps * self.substeps)
        params = {k: p.detach().clone() for k, p in self.model.named_parameters()}
        if cfg.optimizer == "adam":
            return AdamState(params=params, lr=schedule, betas=(cfg.beta1, cfg.beta2), wd=cfg.weight_decay)
        if cfg.optimizer == "momentum":
            return SGDMomentumState(params=params, lr=schedule, wd=cfg.weight_decay, momentum=cfg.beta1)
        if cfg.optimizer == "sharpness":
            return SharpnessState(params=params, lr=schedule, wd=cfg.weight_decay)
        if cfg.optimizer == "newton":
            return NewtonState(params=params, lr=schedule, wd=cfg.weight_decay)
        if cfg.optimizer == "edgeflow":
            return EdgeFlowState(params=params, lr=schedule, wd=cfg.weight_decay)
        if cfg.optimizer == "centralflow":
            return CentralFlowState(params=params, lr=schedule, wd=cfg.weight_decay, compile=self.compile, devices=tuple(self.helper_devices))
        if cfg.optimizer == "heun":
            return HeunState(params=params, lr=schedule, wd=cfg.weight_decay)
        assert cfg.optimizer == "sgd", cfg.optimizer
        return SGDState(params=params, lr=schedule, wd=cfg.weight_decay)

    @cached_property
    def objective(self) -> Objective:
        return Objective(model=self.model, loss_fn=LOSSES[self.cfg.loss], bf16=self.cfg.dtype == "bf16")

    @cached_property
    def magic(self) -> Magic:
        return Magic(objective=self.objective, loader=self.loader, compile=self.compile)

    @cached_property
    def start_state(self) -> OptimState:
        """The state at step cfg.start: `init_state` trained on all the data (saved in DIR/full_model/start.pt)."""
        if self.cfg.start == 0:
            return self.init_state
        path = self.dir / "full_model" / "start.pt"
        if path.exists():
            return self.load_state(path)
        click.echo(f"training the base model to step {self.cfg.start}")
        state = self.magic.train(self.init_state, n_steps=self.steps[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(asdict(state), path)
        return state

    def load_state(self, path: Path) -> OptimState:
        """A state saved as `asdict(state)` (on CPU)."""
        saved = torch.load(path, weights_only=True)
        if "compile" in saved:
            saved["compile"] = self.compile
        if "devices" in saved:
            saved["devices"] = tuple(self.helper_devices)
        return type(self.init_state)(**saved)

    def checkpoints(self) -> dict[int, Path]:
        """The optimizer steps strictly inside the training with a saved state (DIR/full_model/checkpoints), in order"""
        start, n_steps = self.steps
        paths = {int(path.stem): path for path in (self.dir / "full_model" / "checkpoints").glob("*.pt")}
        return {t: paths[t] for t in sorted(paths) if start < t < start + n_steps}

    def eval_losses(self, model_dir: Path, eval_sets: list[Tensor], extra_step: bool = False) -> Tensor:
        """The loss of the model model_dir/final.pt (DIR/full_model or DIR/retrained/<run>) on each eval set (test indices)
        [n_fns], after one more training step if `extra_step`: read from model_dir/losses.json where saved there, else
        computed from the state and saved there."""
        path = model_dir / "losses.json"
        saved = json.loads(path.read_text()) if path.exists() else {}
        cache = saved.setdefault("extra_step" if extra_step else "final", {})
        keys = [eval_set_key(idx) for idx in eval_sets]
        missing = [i for i, k in enumerate(keys) if k not in cache]
        if missing:
            state_path = model_dir / "final.pt"
            if not state_path.exists():
                raise click.UsageError(f"no state {state_path} (pruned?) and no saved losses of {len(missing)} eval sets in {path}: "
                                       f"run `python main.py {'train' if model_dir.name == 'full_model' else 'retrain'}` again")
            state = self.load_state(state_path).apply(lambda t: t.to(self.device))
            if extra_step:
                start, n_steps = self.steps
                state = self.magic.train(state, n_steps=1, start=start + n_steps, pbar=False)
            params = {k: t.to(self.device) for k, t in state.params.items()}
            for i in missing:
                cache[keys[i]] = self.objective.eval_loss(params, self.test_loader.iter_batches(eval_sets[i]))[0].item()
            path.write_text(json.dumps(saved, indent=1))
        return torch.tensor([cache[k] for k in keys])


def eval_set_key(indices: Tensor) -> str:
    """The key of an eval set (test indices [n]) in losses.json: <n>:<hash of the sorted indices>"""
    return f"{len(indices)}:{hashlib.sha1(indices.sort().values.to(torch.int64).cpu().numpy().tobytes()).hexdigest()[:16]}"


def has_model(run_dir: Path) -> bool:
    """Whether a retrained model's dir has its state or saved losses (`eval_losses`)"""
    return (run_dir / "final.pt").exists() or (run_dir / "losses.json").exists()


OLD_NAMES = {"beta": "d_weights", "betas": "d_weights", "delta": "d_state"}


def load_renamed(path: Path) -> dict:
    """torch.load of a dict saved by `magic`, with the old names of its keys and metrics replaced (`OLD_NAMES`).
    TODO: delete it and `OLD_NAMES` (plain torch.load instead) once no results from before 2026-11-19 are read."""
    saved = torch.load(path, weights_only=True)
    return {"/".join(OLD_NAMES.get(part, part) for part in key.split("/")): value for key, value in saved.items()}


def retrained_dirname(kind: str, alpha: float, index: int) -> str:
    """The subdir of DIR/retrained of a retrained model"""
    return f"{kind}_{alpha:g}_{index}"




exp_dir = click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
device_option = click.option("--device", type=str, default="cpu", show_default=True, help="cpu, cuda, or a number of GPUs (8): retrain trains the models in a process per GPU; centralflow splits LOBPCG's HVPs and, in the magic backward, the pairs of critical eigenvectors among them")
compile_option = click.option("--compile/--no-compile", default=False, help="torch.compile the model forward of the training passes (train: only without second derivatives)")
show_option = click.option("--show/--no-show", default=True, help="show the plots")


@click.group(help=__doc__)
def main():
    pass


@main.command(help="Create the experiment DIR and save its settings to DIR/config.json.")
@exp_dir
@click.option("--dataset", type=click.Choice(list(DATASETS)), default="cifar10", show_default=True)
@click.option("--model", type=click.Choice(list(MODELS)), default="cnn", show_default=True)
@click.option("--loss", type=click.Choice(list(LOSSES)), default="ce", show_default=True)
@click.option("--width", type=int, default=None, help="of the model (mlp: hidden, cnn: channels of the 1st stage, vit: embedding; default: 512 / 32 / 96; cnn 6 ~ 11k params)")
@click.option("--depth", type=int, default=None, help="of the model (mlp: hidden layers, cnn: blocks per stage, vit: blocks; default: 2 / 1 / 3)")
@click.option("--init-seed", type=int, default=0, show_default=True)
@click.option("--data-seed", type=int, default=0, show_default=True, help="seed of the batch order (and of the --n-train subset)")
@click.option("--n-train", type=int, default=None, help="train on a random subset of this many examples (default: all)")
@click.option("--n-classes", type=int, default=None, help="keep only the first this many classes, train and test (default: all)")
@click.option("--batch-size", type=int, default=128, show_default=True)
@click.option("--optimizer", type=click.Choice(["sgd", "momentum", "adam", "sharpness", "newton", "edgeflow", "centralflow", "heun"]), default="sgd", show_default=True, help="sharpness: SGD with the step size lr * 2 / sharpness; edgeflow: Edge Flow with GD learning rate lr; centralflow: the central flow of GD with learning rate lr (logs its own top eigenvalues); heun: 2-stage Runge-Kutta on the gradient flow")
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


@main.command(help="Train the full model on all the data from step `start`, with the logs (see `magic.TrainLog`), into DIR/full_model: "
              "final.pt, metrics.pt and plots. Skipped if final.pt exists, unless --force; a killed run continues from its last checkpoint (--checkpoint-every). By default only the logs that need no second "
              "derivative (and so allow --compile): the eigenvalues, curvature and tangent are opt-in.")
@exp_dir
@device_option
@compile_option
@click.option("--force/--no-force", default=False, help="retrain (and overwrite) an existing full model")
@click.option("--num-top-eigs", type=int, default=0, show_default=True, help="log the top Hessian eigenvalues (0: don't; sharpness: the step's own for 0 or 1)")
@click.option("--num-neg-eigs", type=int, default=0, show_default=True, help="log the most negative Hessian eigenvalues (0: don't; newton: the step's own for 0 or 1)")
@click.option("--eig-every", type=int, default=10, show_default=True, help="log the eigenvalues and curvature every this many steps")
@click.option("--curvature/--no-curvature", default=False, help="log ||Hv|| / ||v|| and v^T H v / v^T v of the batch Hessian along the gradient, the tangent (if --tangent) and params - short-term EMA (one HVP each)")
@click.option("--tangent/--no-tangent", default=False, help="log the norms of ds_t/ds_0 v for a random v (forward-mode AD through the training)")
@click.option("--eval-every", type=int, default=100, show_default=True, help="log the test loss and metrics every this many steps (0: never)")
@click.option("--n-test", type=int, default=10_000, show_default=True, help="test examples (a fresh random subset at each eval, the whole test set at the last step) the test loss is logged on (0: always the whole test set)")
@click.option("--ema-gamma", type=float, default=16.97, show_default=True, help="also log the loss and metrics of the power-function EMA of the params (Karras et al. 2024) with this exponent: 16.97 averages the last ~5% of the training (0: don't; never for centralflow)")
@click.option("--ema-decay", type=float, default=0.9, show_default=True, help="short-term EMA of the params with this decay, the direction params - EMA of --curvature (0: don't; centralflow: only with --curvature)")
@click.option("--checkpoint-every", type=int, default=0, show_default=True, help="save the state every this many steps (0: don't): a killed train continues from the last one, and magic replays only between them")
@show_option
def train(exp_dir, device, compile, force, num_top_eigs, num_neg_eigs, eig_every, curvature, tangent, eval_every, n_test, ema_gamma, ema_decay, checkpoint_every, show):
    exp = Experiment(exp_dir, device, compile)
    full_dir = exp_dir / "full_model"
    if (full_dir / "final.pt").exists() and not force:
        click.echo(f"{full_dir / 'final.pt'} exists (--force to retrain)")
        return
    if force:
        shutil.rmtree(full_dir / "checkpoints", ignore_errors=True)
        for name in ("final.pt", "resume.pt"):
            (full_dir / name).unlink(missing_ok=True)
    if (full_dir / "resume.pt").exists():
        click.echo(f"continuing from the last checkpoint ({full_dir / 'resume.pt'})")
    logger = Logger(exp.name, step_scale=exp.substeps, lags=(), smooth=exp.cfg.optimizer != "centralflow")
    if exp.cfg.optimizer == "centralflow":  # the flow does not oscillate: the EMAs (of the params, of the plotted metrics) only lag behind
        ema_gamma, ema_decay = 0, ema_decay if curvature else 0
    start, n_steps = exp.steps
    train_log = TrainLog(
        magic=exp.magic, logger=logger, test_loader=exp.test_loader,
        n_top_eigs=num_top_eigs, n_neg_eigs=num_neg_eigs, eig_every=eig_every, curvature=curvature, tangent=tangent,
        eval_every=eval_every, n_test=n_test or None, ema_gamma=ema_gamma or None, ema_decay=ema_decay or None,
        checkpoint_every=checkpoint_every * exp.substeps, save_dir=full_dir,
    )
    final = train_log.train(exp.start_state, n_steps=n_steps, start=start)
    full_dir.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(final), full_dir / "final.pt")
    (full_dir / "resume.pt").unlink(missing_ok=True)
    logger.finish(full_dir / "metrics.pt")
    click.echo(f"saved {full_dir / 'final.pt'} and {full_dir / 'metrics.png'}")
    if show:
        plt.show()
    plt.close("all")


@main.command(help="Train --n-subsets models with the training weights perturbed from step `start` on, into DIR/retrained: dropping a "
              "random --alpha fraction of the data (--kind drop), or adding N(0, alpha) to every weight (--kind randn; same ||w - 1|| as drop), "
              "vmapped together (first-order optimizers), on several GPUs at once with --device <number of GPUs>. "
              "Existing models are skipped.")
@exp_dir
@device_option
@compile_option
@click.option("--kind", type=click.Choice(["drop", "randn"]), default="randn", show_default=True)
@click.option("--alpha", "alphas", type=float, multiple=True, default=[0.05], show_default=True, help="size of the perturbation (repeatable)")
@click.option("--n-subsets", type=int, default=10, show_default=True, help="perturbed models per alpha (indices 0, ..., n-1)")
@click.option("--vmap-limit", type=int, default=None, help="train at most this many models per vmapped pass (default: all at once; 1: not vmapped)")
def retrain(exp_dir, device, compile, kind, alphas, n_subsets, vmap_limit):
    devices = parse_devices(device)
    if len(devices) == 1:
        return retrain_shard(exp_dir, device, compile, kind, alphas, n_subsets, vmap_limit)
    Experiment(exp_dir, devices[0], compile).start_state  # (saved, so that the processes don't each train it)
    context = torch.multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=retrain_shard, args=(exp_dir, d, compile, kind, alphas, n_subsets, vmap_limit, (i, len(devices))))
        for i, d in enumerate(devices)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if failed := [d for d, process in zip(devices, processes) if process.exitcode != 0]:
        raise click.ClickException(f"retrain failed on {', '.join(failed)}")


def retrain_shard(exp_dir, device, compile, kind, alphas, n_subsets, vmap_limit, shard: tuple[int, int] = (0, 1)):
    """`retrain` on one device, of the runs i mod n of `shard` = (i, n)"""
    exp = Experiment(exp_dir, device, compile)
    start, n_steps = exp.steps
    all_runs = {(alpha, i): exp_dir / "retrained" / retrained_dirname(kind, alpha, i) for alpha in alphas for i in range(n_subsets)}  # -> dir
    all_runs = dict(list(all_runs.items())[shard[0] :: shard[1]])
    runs = {key: d for key, d in all_runs.items() if not (d / "final.pt").exists()}
    limit = 1 if isinstance(exp.init_state, SecondOrderState) else vmap_limit or len(runs) or 1  # (a second-order step can't be vmapped)
    if runs:
        click.echo(f"training {len(runs)} perturbed models" + (f" in chunks of {limit}" if limit < len(runs) else "") + (f" on {device}" if shard[1] > 1 else ""))
    for c in range(0, len(runs), limit):
        chunk = list(runs.items())[c:c + limit]
        weights = [exp.train_ds.weights(kind, alpha, index=i) for (alpha, i), _ in chunk]
        vmapped = len(chunk) > 1
        final = exp.magic.train(exp.start_state, n_steps=n_steps, start=start, weights=torch.stack(weights) if vmapped else weights[0])
        for r, (((alpha, _), d), w) in enumerate(zip(chunk, weights)):
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"weights": w, "kind": kind, "alpha": alpha}, d / "perturbation.pt")
            torch.save(asdict(final.apply(lambda t: t[r]) if vmapped else final), d / "final.pt")


def plot_examples(exp: Experiment, subset_dir: Path, k: int = 5):
    """Plot the eval set of `subset_dir` (test images) and the `k` training images with the lowest and highest df/dw
    (upweighting / dropping them lowers f the most), to subset_dir/examples.png; only for image datasets."""
    train_ds, test_ds = exp.train_ds, exp.test_ds
    if not isinstance(train_ds, Images):
        return
    lin = load_renamed(subset_dir / "linearization.pt")
    d_weights = lin["d_weights"]
    order = d_weights.argsort()
    rows = [("eval set (test examples)", test_ds, lin["indices"]),
            (f"most helpful training examples (lowest df/dw: upweighting lowers f)", train_ds, order[:k]),
            (f"most harmful training examples (highest df/dw: dropping lowers f)", train_ds, order[-k:].flip(0))]
    ncols = max(len(idx) for _, _, idx in rows)
    fig = plt.figure(figsize=(1.6 * ncols, 2.3 * len(rows)), layout="constrained")
    for (title, ds, idx), sub in zip(rows, fig.subfigures(len(rows), 1)):
        sub.suptitle(title, x=0.01, ha="left", fontsize=9)
        axes = sub.subplots(1, ncols, squeeze=False)[0]
        for ax in axes:
            ax.axis("off")
        for ax, i in zip(axes, idx.tolist()):
            img = ds.X[ds.ids[i]].permute(1, 2, 0).numpy()
            ax.imshow(img.squeeze(-1) if img.shape[-1] == 1 else img, cmap="gray" if img.shape[-1] == 1 else None)
            label = f"{ds.classes[ds.labels[i]]} #{ds.ids[i].item()}"
            ax.set_title(label if ds is test_ds else f"{label}\n{d_weights[i].item():.3g}", fontsize=7)
    fig_title(fig, f"MAGIC df/dw, f = {lin['output'].item():.4f}; {exp.name}")
    fig.savefig(subset_dir / "examples.png", dpi=150)


@main.command(help="Compute df/dw (MAGIC) for the loss f of the trained full model (`train` first) on each eval set, "
              "for the training weights of the steps from `start` on. "
              "Saves to DIR/magic (linearization.pt per eval set; metrics.pt and "
              "plots of f(s_t), the norms of df/ds_t and df/dw_t; per eval set, examples.png: its images and the training images with the "
              "lowest / highest df/dw); existing eval sets are skipped. The backward goes from each checkpoint of `train` to the next, "
              "and a killed run continues from the last one it reached.")
@exp_dir
@device_option
@compile_option
@click.option("--n-test", type=int, default=1, show_default=True, help="test examples in each eval set (a fixed random subset): f is their mean loss")
@click.option("--n-subsets", type=int, default=1, show_default=True, help="eval sets (indices 0, ..., n-1)")
@click.option("--passes", type=int, default=2, show_default=True, help="re-trainings to reconstruct the states between two checkpoints in reverse (memory: passes * steps^(1/passes) states)")
@show_option
def magic(exp_dir, device, compile, n_test, n_subsets, passes, show):
    exp = Experiment(exp_dir, device, compile)
    start, n_steps = exp.steps
    final_path = exp_dir / "full_model" / "final.pt"
    if not final_path.exists():
        raise click.UsageError(f"no trained model {final_path}: run `python main.py train` first")
    magic_dir = exp_dir / "magic"
    dirs = {i: magic_dir / f"{n_test}_examples_{i}" for i in range(n_subsets)}  # index -> dir
    todo = {i: d for i, d in dirs.items() if not (d / "linearization.pt").exists()}
    if todo:
        indices = [exp.test_ds.subset(n_test, index=i) for i in todo]  # the test examples of each eval set
        click.echo(f"backpropagating {len(todo)} eval sets")
        logger = Logger(f"MAGIC: df/ds_t and df/dw_t; {exp.name}", smooth=False, split_norms=True, step_scale=exp.substeps)
        eval_sets = [exp.test_ds.examples(idx) for idx in indices]
        final = exp.load_state(final_path)
        outputs, f_grads = exp.objective.eval_grads(to_device(final.params, exp.device), eval_sets)  # f(s_T) [#fns]
        d_state = final.cotangent(f_grads)  # df/ds_T
        d_weights = torch.zeros(len(todo), len(exp.train_ds))  # df/dw [#fns, N]
        checkpoints = exp.checkpoints()
        bounds = [start, *checkpoints, start + n_steps]
        resume_path, names = magic_dir / "resume.pt", [d.name for d in todo.values()]
        if resume_path.exists() and (resume := load_renamed(resume_path))["eval_sets"] == names:
            click.echo(f"continuing from step {resume['step']} ({resume_path})")
            d_state, d_weights = d_state.unflatten(resume["d_state"]), resume["d_weights"]
            bounds = [t for t in bounds if t <= resume["step"]]
            logger.restore(load_renamed(magic_dir / "metrics.pt"))
        for seg_start, seg_stop in reversed(list(zip(bounds, bounds[1:]))):
            init_state = exp.load_state(checkpoints[seg_start]) if seg_start in checkpoints else exp.start_state
            d_state, seg_d_weights = backprop_logged(
                exp.magic, init_state, d_state, eval_sets, logger, n_steps=seg_stop - seg_start, start=seg_start, passes=passes
            )
            d_weights += seg_d_weights.cpu()
            magic_dir.mkdir(parents=True, exist_ok=True)
            logger.finish(magic_dir / "metrics.pt")
            torch.save({"step": seg_start, "eval_sets": names, "d_state": [d.cpu() for d in d_state], "d_weights": d_weights}, resume_path)
        for (i, d), idx, output, fn_d_weights in zip(todo.items(), indices, outputs, d_weights):
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"indices": idx, "output": output.cpu(), "d_weights": fn_d_weights}, d / "linearization.pt")
            click.echo(f"  {n_test} examples #{i}: f = {output.item():.4f}")
        logger.finish(magic_dir / "metrics.pt")
        resume_path.unlink()
    for d in dirs.values():
        plot_examples(exp, d)
    if show:
        plt.show()
    plt.close("all")


@main.command(help="Delete the saved states of DIR (DIR/retrained/*/final.pt, DIR/full_model/{start,final}.pt and its checkpoints, almost all of its "
                   "disk usage) to free disk space, keeping the perturbations, df/dw, metrics, plots, evaluations and the losses "
                   "saved by them (losses.json: the evaluations of the same eval sets still run; others need the states again). "
                   "A DIR without config.json is a dir of experiments: each subdir with a config.json is pruned.")
@exp_dir
@click.option("--keep-full-model/--no-keep-full-model", default=False, help="keep DIR/full_model/{start,final}.pt and its checkpoints")
@click.option("--dry-run/--no-dry-run", default=False, help="only list what would be deleted")
@click.option("--yes/--no-yes", "-y", default=False, help="don't ask for confirmation")
def prune(exp_dir: Path, keep_full_model: bool, dry_run: bool, yes: bool):
    exp_dirs = [exp_dir] if (exp_dir / "config.json").exists() else sorted(d for d in exp_dir.iterdir() if (d / "config.json").exists())
    if not exp_dirs:
        raise click.UsageError(f"no config.json in {exp_dir} or its subdirs")
    paths = []
    for d in exp_dirs:
        paths += sorted((d / "retrained").glob("*/final.pt"))
        if not keep_full_model:
            paths += [p for p in (d / "full_model" / "start.pt", d / "full_model" / "final.pt") if p.exists()]
            paths += sorted((d / "full_model" / "checkpoints").glob("*.pt"))
    size = sum(p.stat().st_size for p in paths)
    if not paths:
        click.echo(f"no saved states in {exp_dir}")
        return
    click.echo(f"{len(paths)} saved states, {size / 2**20:.0f} MB, in {len(exp_dirs)} experiments in {exp_dir}:"
               + "".join(f"\n  {p.relative_to(exp_dir)}" for p in paths[:5]) + (f"\n  ... ({len(paths) - 5} more)" if len(paths) > 5 else ""))
    if dry_run or not (yes or click.confirm("delete them?", default=False)):
        return
    for p in paths:
        p.unlink()
    click.echo(f"deleted {len(paths)} files, {size / 2**20:.0f} MB freed")


if __name__ == "__main__":
    main()
