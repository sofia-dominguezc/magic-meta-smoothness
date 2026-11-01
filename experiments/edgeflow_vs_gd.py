"""
How far does Edge Flow drift from the gradient descent it models? Train GD (SGD with the config's lr) and Edge Flow
(DIR must have `--optimizer edgeflow`) side by side from DIR/full_model/start.pt (the init if start = 0) on the same
batches, and compare them at every GD step (running statistics from `Magic.train.trace`, no models stored):
  python -m experiments.edgeflow_vs_gd DIR [--eig-every 10] [--eval-every 50] [--n-test 1000] [--device cpu] [--no-show]

Outputs in DIR/edgeflow_vs_gd/ (metrics.pt and metrics.png, panels in GD steps), w = GD, c = the Edge Flow center,
u = its direction, x = its magnitude, ema = the EMA of w (`OptimState.ema_params`):
- loss_center: the batch loss at c and ema; loss_gd: at w and the symmetrized one (c +- x u); test_loss: on --n-test
  examples every --eval-every
- distance: |w - c|, |ema - c|, |w - ema|; travel: |w - w_0|, |c - w_0| (how far the models move at all)
- amplitude_along_u: x, and GD's displacement along u: |(w - c) . u|, |(w - ema) . u|
- drift_orth: |w - c| and |ema - c| orthogonal to u (the drift of the center that isn't the oscillation)
- eig: Edge Flow's sharpness |H u| at c, GD's top Hessian eigenvalue at w (every --eig-every steps), and 2 / lr
- cos_with_u: |cos| of u with GD's gradient at w, and with the top Hessian eigenvector at w
"""

from dataclasses import replace
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from tqdm import tqdm

from main import Experiment
from src.lobpcg import hessian_eigs
from src.logging import Logger
from src.optimizers import EdgeFlowState, LRSchedule, SGDState
from src.utils import dict_zip


def flat(d: dict[str, Tensor]) -> Tensor:
    return torch.cat([t.detach().flatten() for t in d.values()])


def edgeflow_vs_gd(exp_dir: Path, device: str = "cpu", eig_every: int = 10, eval_every: int = 50, n_test: int = 1000) -> dict:
    exp = Experiment(exp_dir, device)
    cfg = exp.cfg
    if cfg.optimizer != "edgeflow":
        raise click.UsageError(f"{exp_dir} has optimizer {cfg.optimizer}, not edgeflow")
    K = EdgeFlowState.SUBSTEPS
    ef_state = exp.start_state
    gd_state = SGDState(params=ef_state.params, lr=LRSchedule(cfg.lr, kind=cfg.lr_schedule, warmup=cfg.warmup, n_steps=cfg.n_steps),
                        wd=cfg.weight_decay)
    gd_magic = replace(exp.magic, loader=replace(exp.loader, repeat=1))  # the same batch t as Edge Flow's substeps K t, ..., K t + K - 1
    start, n_steps = cfg.start, cfg.n_steps - cfg.start
    test_idx = exp.test_ds.subset(min(n_test, len(exp.test_ds)), index=0)
    logger = Logger(f"Edge Flow (K = {K}) vs GD; {exp.name}", smooth=False)
    logger.log_panels = {"amplitude_along_u"}

    ef = exp.magic.train.trace(ef_state, n_steps=n_steps * K, start=start * K, pbar=False)
    gd = gd_magic.train.trace(gd_state, n_steps=n_steps, start=start, pbar=False)
    tt, s_ef = -1, None
    w0 = flat(ef_state.params).to(device)
    eigvec = None  # GD's top eigenvector, warm start
    for t, s_gd, g_gd in tqdm(gd, total=n_steps + 1, desc="edge flow vs gd"):
        while tt < t * K:
            tt, s_ef, _ = next(ef)
        assert s_ef is not None
        with torch.no_grad():
            X, Y, _ = gd_magic.loader.get_batch(min(t, start + n_steps - 1))
            w, ema, c, u = (flat(d) for d in (s_gd.params, s_gd.ema_params, s_ef.params, s_ef.eigvec))
            x = s_ef.x["x"]
            d_wc, d_ec = w - c, ema - c
            proj_wc, proj_ec = d_wc @ u, d_ec @ u
            ema_loss = gd_magic.forward(s_gd.ema_params, X, Y)[0].mean()
            symmetrized = sum(exp.magic.forward({k: p + s * x * v for k, (p, v) in dict_zip(s_ef.params, s_ef.eigvec).items()}, X, Y)[0].mean()
                              for s in (1, -1)) / 2
            logger.log(t, **{"loss_center/center": exp.magic.forward(s_ef.params, X, Y)[0].mean(), "loss_center/ema": ema_loss})
            logger.log(t, **{"loss_gd/gd": gd_magic.forward(s_gd.params, X, Y)[0].mean(), "loss_gd/symmetrized": symmetrized})
            logger.log(t, **{"distance/gd-center": d_wc.norm(), "distance/ema-center": d_ec.norm(), "distance/gd-ema": (w - ema).norm()})
            logger.log(t, **{"travel/gd": (w - w0).norm(), "travel/center": (c - w0).norm()})
            logger.log(t, **{"amplitude_along_u/x": x})
            if t > start:  # (0 at start: log scale)
                logger.log(t, **{"amplitude_along_u/gd-center": proj_wc.abs(), "amplitude_along_u/gd-ema": ((w - ema) @ u).abs()})
            logger.log(t, **{"drift_orth/gd-center": (d_wc - proj_wc * u).norm(), "drift_orth/ema-center": (d_ec - proj_ec * u).norm()})
            logger.log(t, **{"eig/threshold": torch.tensor(2 / s_gd.lr(t)), "eig/edgeflow": s_ef.stats["sharpness"]})  # (nan at start; before GD's: drawn under it)
            if g_gd is not None:
                g = flat(g_gd)
                logger.log(t, **{"cos_with_u/gd_gradient": (g @ u).abs() / g.norm()})
            if eval_every and ((t - start) % eval_every == 0 or g_gd is None):
                for name, params in (("gd", s_gd.params), ("ema", s_gd.ema_params), ("center", s_ef.params)):
                    logger.log(t, **{f"test_loss/{name}": exp.magic.eval_loss(params, exp.test_loader, test_idx)[0]})
        if g_gd is not None and (t - start) % eig_every == 0:
            hvp_fn = gd_magic.hvp_fn(s_gd.params, X, Y)
            eigvec = {k: v.detach().clone() for k, v in s_ef.eigvec.items()} if eigvec is None else eigvec
            eigval, eigvec, _ = hessian_eigs(hvp_fn, eigvec, batched=False)
            logger.log(t, **{"eig/gd": eigval, "cos_with_u/gd_top_eigvec": (flat(eigvec) @ u).abs() / flat(eigvec).norm()})

    out_dir = exp_dir / "edgeflow_vs_gd"
    out_dir.mkdir(exist_ok=True)
    logger.finish(out_dir / "metrics.pt")
    metrics = logger.metrics()
    last = {k: v["value"][-1].item() for k, v in metrics.items() if k.startswith(("distance/", "travel/", "amplitude_along_u/", "loss_"))}
    click.echo(f"at step {start + n_steps}: " + ", ".join(f"{k} {v:.3g}" for k, v in last.items()))
    click.echo(f"saved {out_dir / 'metrics.png'}")
    return metrics


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--eig-every", type=int, default=10, show_default=True, help="GD's top Hessian eigenpair every this many GD steps")
@click.option("--eval-every", type=int, default=50, show_default=True, help="test loss every this many GD steps (0: never)")
@click.option("--n-test", type=int, default=1000, show_default=True, help="test examples of the test loss")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dir, eig_every, eval_every, n_test, device, show):
    edgeflow_vs_gd(exp_dir, device, eig_every, eval_every, n_test)
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
