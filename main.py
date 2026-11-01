"""

  python main.py new DIR [settings]   # create an experiment: save its settings (dataset, model, optimizer, start, ...) to DIR/config.json
  python main.py train DIR [...]      # train the full model on all the data, with the logs (loss, metrics, norms; optionally Hessian eigenvalues, batch sharpness, tangent, EMA, test loss)
  python main.py retrain DIR [...]    # train models with perturbed training weights (data dropped / noised), vmapped together
  python main.py magic DIR [...]      # train on all the data and compute df/dw for each eval set (--variant: Gauss-Newton / skipped steps)
The three are independent (each only needs DIR/config.json); experiments/evaluate.py needs all of them.

Everything saved in DIR is comparable (same config, so same training and same steps `start`-`n_steps` perturbed).
A saved state (*.pt below) is `asdict(OptimState)`, loaded by `Experiment.load_state`:
  DIR/full_model/   start.pt (if start > 0): the state at step `start`, trained once on all the data, from which everything
                    continues; final.pt: the state after training on all the data (the retrained models' reference in
                    evaluate.py: the plain and vmapped passes differ by fp32 rounding, see experiments/vmap_error.py);
                    metrics.pt + plots: the logs of `train` (see `train_logged`)
  DIR/retrained/    one dir per retrained model: final.pt, perturbation.pt = {weights [N], kind, alpha}
  DIR/magic/        one dir per eval set: linearization.pt = {indices [n] (test examples), output f, beta df/dw [N]};
                    metrics.pt + plots: the norms of df/ds_t and df/dw_t of the last `magic` call
  DIR/magic_<v>/    the same with `magic --variant <v>` (Gauss-Newton / skipped steps, see `parse_variant`; evaluated into DIR/evaluation_<v>)
The dir names (<kind>_<alpha>_<index>, <n>_examples_<index>) are labels: nothing is parsed from them.

retrain and magic can be run several times (e.g. with other alphas, more eval sets): existing models / eval sets are skipped;
train skips an existing full model unless --force. experiments/ compares them (experiments/evaluate.py), tunes the lr
(experiments/tune_lr.py; needed if `new` is given no --lr) and runs whole sweeps (experiments/sweep.py).
"""

import json
import re
import click
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property

import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor

from src.data import DATASETS, Dataset, Images, Loader
from src.lobpcg import hessian_eigs
from src.magic import Magic
from src.utils import LOSSES, dict_zip, hvp_from_grads, metrics
from src.models import MODELS
from src.optimizers import AdamState, EdgeFlowState, LRSchedule, NewtonState, OptimState, SecondOrderState, SGDMomentumState, SGDState, SharpnessState
from src.logging import Logger, fig_title, op_norm


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
    optimizer: str = "sgd"  # sgd, momentum, adam, sharpness (SGD with the step size lr / top Hessian eigenvalue), newton or edgeflow
    damping: float = 0.01  # newton only
    lr: float | None = None
    lr_schedule: str = "constant"  # or cosine, after `warmup` linear steps (models.LRSchedule)
    warmup: int = 0
    beta1: float = 0.9  # momentum/adam only
    beta2: float = 0.999  # adam only
    weight_decay: float = 0.0
    n_steps: int = 400  # gradient descent steps (edge flow: EdgeFlowState.SUBSTEPS optimizer steps each, see `Experiment.steps`)
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
        return ds.to(self.device)

    @cached_property
    def test_ds(self) -> Dataset:
        return DATASETS[self.cfg.dataset](train=False).to(self.device)

    @property
    def substeps(self) -> int:
        """Optimizer steps per step of the config"""
        return EdgeFlowState.SUBSTEPS if self.cfg.optimizer == "edgeflow" else 1

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
        width = {} if self.cfg.width is None else {"width": self.cfg.width}
        return MODELS[self.cfg.model](self.train_ds.in_shape, self.train_ds.out_dim, **width).to(self.device)

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
            return NewtonState(params=params, lr=schedule, wd=cfg.weight_decay, damping=cfg.damping)
        if cfg.optimizer == "edgeflow":
            return EdgeFlowState(params=params, lr=schedule, wd=cfg.weight_decay)
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
        state = self.magic.train(self.init_state, n_steps=self.steps[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(asdict(state), path)
        return state

    def load_state(self, path: Path) -> OptimState:
        """A state saved as `asdict(state)` (on CPU)."""
        return type(self.init_state)(**torch.load(path, weights_only=True))


def eig_cos(d: dict[str, Tensor], eigvecs: dict[str, Tensor]) -> Tensor:
    """|cos| of the direction `d` {name: [*shape]} with each of the k eigenvectors {name: [k, *shape]} -> [k];
    a batch of directions {name: [n, *shape]} -> [n, k]"""
    name, vec = next(iter(eigvecs.items()))
    batched = d[name].ndim == vec.ndim
    D = torch.cat([d[name].flatten(1 if batched else 0).reshape(-1 if batched else 1, d[name][0].numel() if batched else d[name].numel()) for name in eigvecs], 1)  # [n, N]
    V = torch.cat([vec.flatten(1) for vec in eigvecs.values()], 1)  # [k, N]
    cos = (D @ V.T) / (D.norm(dim=1)[:, None] * V.norm(dim=1)[None])
    return cos.abs() if batched else cos.abs()[0]


def train_logged(
    magic: Magic, init_state: OptimState, *, n_steps: int, start: int = 0, test_loader: Loader, logger: Logger,
    n_top_eigs: int = 0, n_neg_eigs: int = 0, eig_every: int = 10, batch_sharpness: bool = False, tangent: bool = False,
    eval_every: int = 100, n_test: int | None = None, ema_decay: float | None = 0.9,
) -> OptimState:
    """Train the full model (`Magic.train` with its own logs: loss, metrics, op norms of the params and grads, the step's stats)
    with more logs from the states and gradients it yields. By default only the ones that need no second derivative:
    - "<name>/ema": the loss and metrics with the state's EMA of the params (`ema_decay` sets `OptimState.ema_decay`);
      "ema_eig/i": |cos| of params - EMA with the eigenvectors
    - "test_<name>": on `n_test` random test examples every `eval_every` steps (the whole test set at the last step)
    - "lr" if scheduled
    With second derivatives (`create_graph`: the forward isn't compiled):
    - "eig/i": top (i > 0) / most negative (i < 0) eigenvalues of the Hessian of the batch loss (unscaled: not of the preconditioned
      Hessian), "eigvec/i" their norms, every `eig_every` steps. The eigenpair a 2nd-order step computes (sharpness: the top, newton:
      the most negative) is logged at every step; LOBPCG only runs for 2 or more eigenpairs of that sign.
      "eig/threshold": the stability threshold on the same scale, `stability_threshold` / step size P (when P is a scalar: not adam)
    - "eig/batch_sharpness": g^T H g / g^T g of the batch gradient g and Hessian H (a stability estimate for small batches),
      every `eig_every` steps
    - "tangent": the norms of ds_t/ds_0 v for a random v, propagated by forward-mode AD through the training (the loss gradient
      and the step), for any optimizer; "tangent_eig/i": its |cos| with the eigenvectors
    Returns the trained state."""
    second_order = {1: "sharpness"} if isinstance(init_state, (SharpnessState, EdgeFlowState)) \
        else {-1: "min_eig"} if isinstance(init_state, NewtonState) \
        else {}
    ks = {sign: k for sign, k in ((1, n_top_eigs), (-1, n_neg_eigs)) if k > 1 or k == 1 and sign not in second_order}  # sign -> eigenpairs by LOBPCG
    stored = {sign: name for sign, name in second_order.items() if sign not in ks}  # sign -> the stat of the eigenpair the step keeps (aux `eigvec`)
    eos = init_state.stability_threshold
    warm: dict[int, dict[str, Tensor]] = {}  # sign -> eigenvectors [k, *shape] per param, LOBPCG's warm start
    if ema_decay is not None:
        init_state = replace(init_state, ema_decay=ema_decay)
    prev_directions: dict[str, dict[str, Tensor] | None] = {}  # of the previous step, for the stored eigenpair

    v: OptimState | None = None  # the tangent ds_t/ds_0 v (plain tensors)
    hook = None
    if tangent:
        g = torch.Generator().manual_seed(start)
        v = init_state.apply(lambda s: torch.randn(s.shape, generator=g, dtype=s.dtype).to(magic.device))
        v = v.apply(lambda s: s / op_norm(s, n_iter=10)[0])
        dual_level: list[int] = []

        def hook(state: OptimState) -> OptimState:
            """Between a step and the next forward: read the tangent the step propagated to the new state (s_{t+1} as a
            dual of the level entered before step t's forward), close that level, and attach it to the detached state in
            a new level, for the next forward and step."""
            nonlocal v
            assert v is not None
            if dual_level:
                tangents = [fwAD.unpack_dual(s).tangent for s in state]
                # detach: else the tangent keeps the step's graph alive, and with
                # create_graph the whole double-backward graph, growing every step
                v = state.unflatten([torch.zeros_like(s) if tan is None else tan.detach() for s, tan in zip(state, tangents)])
                fwAD.exit_dual_level()
                dual_level.pop()
            state = state.apply(lambda s: s.detach())
            dual_level.append(fwAD.enter_dual_level())
            return state.unflatten([fwAD.make_dual(s, tan) for s, tan in zip(state, v)])

    def log_eigs(t: int, ids: list[int], eigvals: Tensor, eigvecs: dict[str, Tensor], directions: dict[str, dict[str, Tensor] | None]):
        """Log "eig/i", the norms "eigvec/i" and the |cos| with `directions` of the eigenpairs `ids` (eigvecs: [k, *shape] per param)"""
        logger.log(t, **{f"eig/{i}": lam for i, lam in zip(ids, eigvals.unbind())})
        for j, i in enumerate(ids):
            logger.log_norms(t, f"eigvec/{i}", {name: vec[j] for name, vec in eigvecs.items()})
        for name, d in directions.items():
            if d is not None:
                logger.log(t, **{f"{name}/{i}": c for i, c in zip(ids, eig_cos(d, eigvecs).unbind())})

    def log_threshold(t: int, state: OptimState):
        """"eig/threshold" = stability threshold / step size, if the step size is a scalar. A second-order state's `precond`
        uses the stats of the step that produced it (step t - 1, at s_{t-1}): logged at t - 1, like its eigenpair"""
        if isinstance(state, SecondOrderState):
            if t == start:
                return  # (no stats yet)
            t -= 1
        P = next(iter(state.precond(t).values()))
        if not isinstance(P, Tensor) or P.ndim == 0:
            logger.log(t, **{"eig/threshold": torch.as_tensor(eos / P, dtype=torch.get_default_dtype())})

    create_graph = bool(ks) or batch_sharpness
    try:
        trace = magic.train.trace(init_state, n_steps=n_steps, start=start, log=logger, create_graph=create_graph, hook=hook)
        for t, state, grads in trace:
            with fwAD._set_fwd_grad_enabled(False):  # the logging computes no tangents (the state and grads are duals if `tangent`)
                params = {k: p.detach() for k, p in state.params.items()}
                X, Y, _ = magic.loader.get_batch(t)
                if state.lr.kind != "constant" or state.lr.warmup:
                    logger.log(t, lr=torch.tensor(state.lr(t)))

                ema = state.ema_params if ema_decay is not None else None
                if ema is not None:
                    with torch.no_grad():
                        losses, out = magic.forward(ema, X, Y)
                        batch_metrics = {"loss": losses.mean(), **{k: m.mean() for k, m in metrics(out, Y).items()}}
                        logger.log(t, **{f"{k}/ema": m for k, m in batch_metrics.items()})

                if v is not None:
                    logger.log_norms(t, "tangent", v)

                directions = {"tangent_eig": None if v is None else v.params,
                              "ema_eig": None if ema is None or t == start else {k: p - e for k, (p, e) in dict_zip(params, ema).items()}}
                if stored and t > start:  # the eigenpair step t - 1 computed at s_{t-1}
                    assert isinstance(state, (SharpnessState, NewtonState, EdgeFlowState))
                    for sign, name in stored.items():
                        log_eigs(t - 1, [sign], state.stats[name][None], {k: u[None] for k, u in state.eigvec.items()}, prev_directions)
                    log_threshold(t, state)
                eig_step = grads is not None and (t - start) % eig_every == 0
                if ks and eig_step and t > start + eig_every:
                    hvp_fn = hvp_from_grads(grads, state.params)
                    for sign, k in ks.items():
                        if sign not in warm:  # random start
                            g = torch.Generator().manual_seed(t)
                            warm[sign] = {name: torch.randn(k, *p.shape, generator=g, dtype=p.dtype).to(p.device) for name, p in params.items()}
                        eigvals, warm[sign], n_iter = hessian_eigs(hvp_fn, warm[sign], None, sign)
                        logger.log(t, **{f"lobpcg_iters/{sign}": torch.tensor(n_iter)})
                        log_eigs(t, [sign * (i + 1) for i in range(k)], eigvals, warm[sign], directions)
                    if not stored:
                        log_threshold(t, state)
                if batch_sharpness and eig_step:
                    assert grads is not None
                    g_ = {k: g.detach() for k, g in grads.items()}
                    Hg = hvp_from_grads(grads, state.params)(g_)
                    gHg = torch.stack([(g * hg).sum() for g, hg in dict_zip(g_, Hg).values()]).sum()
                    gg = torch.stack([g.pow(2).sum() for g in g_.values()]).sum()
                    logger.log(t, **{"eig/batch_sharpness": gHg / gg})
                    if not ks and not stored:
                        log_threshold(t, state)
                prev_directions = directions

                if eval_every and ((t - start) % eval_every == 0 or grads is None):
                    test_ds = test_loader.dataset
                    indices = torch.arange(len(test_ds)) if n_test is None or grads is None else test_ds.subset(n_test, index=t)
                    for p, suffix in ((params, ""), (ema, "/ema")):
                        if p is not None:
                            loss, mean_metrics = magic.eval_loss(p, test_loader, indices)
                            logger.log(t, **{f"test_{k}{suffix}": m for k, m in {"loss": loss, **mean_metrics}.items()})
    finally:
        if tangent and dual_level:  # (also on an error: a level left open breaks the next forward-mode run)
            fwAD.exit_dual_level()
    return state.apply(lambda s: s.detach().cpu())


HESSIAN = "hessian"  # the default `magic` variant


def parse_variant(variant: str, cfg: ExperimentConfig) -> tuple[int | None, int | None]:
    """A `magic` variant -> (gn_below, skip_below) of `Magic.backprop`. The variant is "hessian" (the plain backward), or
    tokens joined by "_": "gn" (the Gauss-Newton matrix at every step), "gn<T>" (at the steps t < T), "skip<T>" (the steps
    t < T contribute nothing), e.g. "gn300_skip100": Hessian for t >= 300, Gauss-Newton for 100 <= t < 300, nothing below."""
    gn_below = skip_below = None
    if variant != HESSIAN:
        for token in variant.split("_"):
            m = re.fullmatch(r"(gn|skip)(\d*)", token)
            if m is None or m[1] == "skip" and not m[2]:
                raise click.BadParameter(f"{token!r} in variant {variant!r}: expected hessian, gn, gn<T> or skip<T>")
            if m[1] == "gn":
                gn_below = int(m[2]) if m[2] else cfg.n_steps
            else:
                skip_below = int(m[2])
    return gn_below, skip_below


def magic_dirname(variant: str) -> str:
    """The subdir of DIR with the `magic` outputs: "magic" for the Hessian, else "magic_<variant>" (evaluated into "evaluation[_<variant>]")"""
    return "magic" if variant == HESSIAN else f"magic_{variant}"


variant_option = click.option("--variant", type=str, default=HESSIAN, show_default=True,
                              help="hessian, gn, gn<T>, skip<T> or e.g. gn300_skip100: the Gauss-Newton matrix in the backward of the steps t < T "
                                   "(first-order optimizers), steps t < T skipped (see `parse_variant`); into DIR/magic_<variant>")


exp_dir = click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
device_option = click.option("--device", type=str, default="cpu", show_default=True)
compile_option = click.option("--compile/--no-compile", default=False, help="torch.compile the model forward of the training passes (train: only without second derivatives)")
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
@click.option("--optimizer", type=click.Choice(["sgd", "momentum", "adam", "sharpness", "newton", "edgeflow"]), default="sgd", show_default=True, help="sharpness: SGD with the step size lr * 2 / sharpness; edgeflow: Edge Flow with GD learning rate lr")
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


@main.command(help="Train the full model on all the data from step `start`, with the logs (see `train_logged`), into DIR/full_model: "
              "final.pt, metrics.pt and plots. Skipped if final.pt exists, unless --force. By default only the logs that need no second "
              "derivative (and so allow --compile): the eigenvalues, batch sharpness and tangent are opt-in.")
@exp_dir
@device_option
@compile_option
@click.option("--force/--no-force", default=False, help="retrain (and overwrite) an existing full model")
@click.option("--num-top-eigs", type=int, default=0, show_default=True, help="log the top Hessian eigenvalues (0: don't; sharpness: the step's own for 0 or 1)")
@click.option("--num-neg-eigs", type=int, default=0, show_default=True, help="log the most negative Hessian eigenvalues (0: don't; newton: the step's own for 0 or 1)")
@click.option("--eig-every", type=int, default=10, show_default=True, help="log the eigenvalues and batch sharpness every this many steps")
@click.option("--batch-sharpness/--no-batch-sharpness", default=False, help="log g^T H g / g^T g of the batch gradient (in the eig panel)")
@click.option("--tangent/--no-tangent", default=False, help="log the norms of ds_t/ds_0 v for a random v (forward-mode AD through the training)")
@click.option("--eval-every", type=int, default=100, show_default=True, help="log the test loss and metrics every this many steps (0: never)")
@click.option("--n-test", type=int, default=1000, show_default=True, help="test examples (a fresh random subset at each eval, the whole test set at the last step) the test loss is logged on (0: always the whole test set)")
@click.option("--ema-decay", type=float, default=0.9, show_default=True, help="also log the loss and metrics of an EMA of the params with this decay, and its |cos| with the Hessian eigenvectors (0: don't)")
@show_option
def train(exp_dir, device, compile, force, num_top_eigs, num_neg_eigs, eig_every, batch_sharpness, tangent, eval_every, n_test, ema_decay, show):
    exp = Experiment(exp_dir, device, compile)
    cfg = exp.cfg
    full_dir = exp_dir / "full_model"
    if (full_dir / "final.pt").exists() and not force:
        click.echo(f"{full_dir / 'final.pt'} exists (--force to retrain)")
        return
    logger = Logger(exp.name, step_scale=exp.substeps, lags=())
    start, n_steps = exp.steps
    final = train_logged(
        exp.magic, exp.start_state, n_steps=n_steps, start=start, test_loader=exp.test_loader, logger=logger,
        n_top_eigs=num_top_eigs, n_neg_eigs=num_neg_eigs, eig_every=eig_every, batch_sharpness=batch_sharpness, tangent=tangent,
        eval_every=eval_every, n_test=n_test or None, ema_decay=ema_decay or None,
    )
    full_dir.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(final), full_dir / "final.pt")
    logger.finish(full_dir / "metrics.pt")
    click.echo(f"saved {full_dir / 'final.pt'} and {full_dir / 'metrics.png'}")
    if show:
        plt.show()
    plt.close("all")


@main.command(help="Train --n-subsets models with the training weights perturbed from step `start` on, into DIR/retrained: dropping a "
              "random --alpha fraction of the data (--kind drop), or adding N(0, alpha) to every weight (--kind randn; same ||w - 1|| as drop), "
              "vmapped together. Existing models are skipped.")
@exp_dir
@device_option
@compile_option
@click.option("--kind", type=click.Choice(["drop", "randn"]), default="randn", show_default=True)
@click.option("--alpha", "alphas", type=float, multiple=True, default=[0.05], show_default=True, help="size of the perturbation (repeatable: all the models in one vmapped pass)")
@click.option("--n-subsets", type=int, default=10, show_default=True, help="perturbed models per alpha (indices 0, ..., n-1)")
@click.option("--vmap-limit", type=int, default=None, help="train at most this many models per vmapped pass (default: all at once)")
def retrain(exp_dir, device, compile, kind, alphas, n_subsets, vmap_limit):
    exp = Experiment(exp_dir, device, compile)
    start, n_steps = exp.steps
    all_runs = {(alpha, i): exp_dir / "retrained" / f"{kind}_{alpha:g}_{i}" for alpha in alphas for i in range(n_subsets)}  # -> dir
    runs = {key: d for key, d in all_runs.items() if not (d / "final.pt").exists()}
    for (alpha, i), d in all_runs.items():  # (a run killed between the two saves of an earlier version: the weights are deterministic)
        if d not in runs.values() and not (d / "perturbation.pt").exists():
            torch.save({"weights": exp.train_ds.weights(kind, alpha, index=i), "kind": kind, "alpha": alpha}, d / "perturbation.pt")
    weights = [exp.train_ds.weights(kind, alpha, index=i) for alpha, i in runs]
    vmap = not isinstance(exp.init_state, SecondOrderState)  # (a second-order step can't be vmapped: one model at a time)
    limit = (vmap_limit or len(weights) or 1) if vmap else 1
    if weights:
        click.echo(f"training {len(weights)} perturbed models" + (f" in chunks of {limit}" if limit < len(weights) else ""))
    for c in range(0, len(weights), limit):
        chunk = torch.stack(weights[c:c + limit]) if vmap else weights[c]
        final = exp.magic.train(exp.start_state, n_steps=n_steps, start=start, weights=chunk)
        for r, (((alpha, _), d), w) in enumerate(list(zip(runs.items(), weights))[c:c + limit]):
            d.mkdir(parents=True, exist_ok=True)
            torch.save(asdict(final.apply(lambda t: t[r]) if vmap else final), d / "final.pt")
            torch.save({"weights": w, "kind": kind, "alpha": alpha}, d / "perturbation.pt")  # (together: a kill leaves no half-saved model)


def plot_examples(exp: Experiment, subset_dir: Path, k: int = 5):
    """Plot the eval set of `subset_dir` (test images) and the `k` training images with the lowest and highest df/dw
    (upweighting / dropping them lowers f the most), to subset_dir/examples.png; only for image datasets."""
    train_ds, test_ds = exp.train_ds, exp.test_ds
    if not isinstance(train_ds, Images):
        return
    lin = torch.load(subset_dir / "linearization.pt")
    beta = lin["beta"]
    order = beta.argsort()
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
            ax.set_title(label if ds is test_ds else f"{label}\n{beta[i].item():.3g}", fontsize=7)
    fig_title(fig, f"MAGIC df/dw, f = {lin['output'].item():.4f}; {exp.name}")
    fig.savefig(subset_dir / "examples.png", dpi=150)


@main.command(help="Train on all the data and compute df/dw (MAGIC) for the loss f on each eval set after training, "
              "for the training weights of the steps from `start` on. Saves to DIR/magic (linearization.pt per eval set; metrics.pt and "
              "plots of f(s_t), the norms of df/ds_t and df/dw_t; per eval set, examples.png: its images and the training images with the "
              "lowest / highest df/dw); existing eval sets are skipped.")
@exp_dir
@device_option
@compile_option
@click.option("--n-test", type=int, default=1, show_default=True, help="test examples in each eval set (a fixed random subset): f is their mean loss")
@click.option("--n-subsets", type=int, default=1, show_default=True, help="eval sets (indices 0, ..., n-1)")
@click.option("--passes", type=int, default=2, show_default=True, help="re-trainings to reconstruct the states in reverse (memory: passes * steps^(1/passes) states)")
@click.option("--vmap-limit", type=int, default=None, help="backpropagate at most this many eval sets at a time, a replay each (default: all at once; only the first chunk is logged)")
@variant_option
@show_option
def magic(exp_dir, device, compile, n_test, n_subsets, passes, vmap_limit, variant, show):
    exp = Experiment(exp_dir, device, compile)
    cfg = exp.cfg
    gn_below, skip_below = (None if s is None else s * exp.substeps for s in parse_variant(variant, cfg))
    start, n_steps = exp.steps
    magic_dir = exp_dir / magic_dirname(variant)
    dirs = {i: magic_dir / f"{n_test}_examples_{i}" for i in range(n_subsets)}  # index -> dir
    todo = {i: d for i, d in dirs.items() if not (d / "linearization.pt").exists()}
    if todo:
        indices = [exp.test_ds.subset(n_test, index=i) for i in todo]  # the test examples of each eval set
        limit = vmap_limit or len(todo)
        click.echo(f"backpropagating {len(todo)} eval sets" + (f" in chunks of {limit}" if limit < len(todo) else ""))
        logger = Logger(f"MAGIC{'' if variant == HESSIAN else f' ({variant})'}: df/ds_t and df/dw_t; {exp.name}", smooth=False, split_norms=True, step_scale=exp.substeps)
        eval_sets = [exp.test_ds.examples(idx) for idx in indices]
        outputs, betas = [], []  # per chunk: f(s_T) [#fns], df/dw [#fns, N]
        for c in range(0, len(eval_sets), limit):
            output, beta = exp.magic.backprop(
                exp.start_state, n_steps=n_steps, start=start, eval_sets=eval_sets[c:c + limit],
                passes=passes, gn_below=gn_below, skip_below=skip_below, log=logger if c == 0 else None,
            )
            outputs.append(output)
            betas.append(beta)
        for (i, d), idx, output, beta in zip(todo.items(), indices, torch.cat(outputs), torch.cat(betas)):
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"indices": idx, "output": output.cpu(), "beta": beta.cpu()}, d / "linearization.pt")
            click.echo(f"  {n_test} examples #{i}: f = {output.item():.4f}")
        logger.finish(magic_dir / "metrics.pt")
    for d in dirs.values():
        plot_examples(exp, d)
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
