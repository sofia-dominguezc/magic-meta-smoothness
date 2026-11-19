"""
How far does Edge Flow (or Heun) drift from the gradient descent it models? Train GD (SGD with the config's lr) and the
config's optimizer (DIR must have `--optimizer edgeflow` or `heun`) side by side from DIR/full_model/start.pt (the init if
start = 0) on the same batches, and compare them at every GD step (running statistics from `Magic.train.trace`, no models stored):
  python -m experiments.edgeflow_vs_gd DIR [--eig-every 10] [--eval-every 50] [--n-test 1000] [--device cpu] [--no-show]

Outputs in DIR/<optimizer>_vs_gd/ (metrics.pt and metrics.png, panels in GD steps), w = GD, c = the other model (the Edge
Flow center, or Heun's params):
- loss_center / loss_heun: the batch loss at c; loss_gd: at w; test_loss: on --n-test examples every --eval-every
- distance: |w - c|; travel: |w - w_0|, |c - w_0| (how far the models move at all)
- eig: the top Hessian eigenvalue of the batch loss at w (every --eig-every steps), and 2 / lr
Edge Flow only (u = its direction, x = its magnitude):
- loss_gd/symmetrized: the batch loss at c +- x u
- amplitude_along_u: Edge Flow's amplitude, and GD's displacement along u: |(w - c) . u|
- drift_orth: |w - c| orthogonal to u (the drift of the center that isn't the oscillation)
- eig/edgeflow: Edge Flow's sharpness |H u| at c
- cos_with_u: |cos| of u with GD's gradient at w, and with the top Hessian eigenvector at w
Heun only:
- eig/heun: the top Hessian eigenvalue at c (every --eig-every steps)
- cos_with_gd_eigvec: |cos| of GD's top Hessian eigenvector with w - c (is GD's distance to Heun its oscillation?), and
  with GD's gradient
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
from src.optimizers import LRSchedule, SGDState
from src.utils import dict_zip


def flat(d: dict[str, Tensor]) -> Tensor:
    return torch.cat([t.detach().flatten() for t in d.values()])


def random_like(d: dict[str, Tensor]) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(0)
    return {k: torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype) for k, p in d.items()}


def edgeflow_vs_gd(exp_dir: Path, device: str = "cpu", eig_every: int = 10, eval_every: int = 50, n_test: int = 1000) -> dict:
    exp = Experiment(exp_dir, device)
    cfg = exp.cfg
    if cfg.optimizer not in ("edgeflow", "heun"):
        raise click.UsageError(f"{exp_dir} has optimizer {cfg.optimizer}, not edgeflow or heun")
    is_edgeflow = cfg.optimizer == "edgeflow"
    other = "center" if is_edgeflow else "heun"  # the name of the compared model in the logs
    K = exp.substeps
    ef_state = exp.start_state
    gd_state = SGDState(params=ef_state.params, lr=LRSchedule(cfg.lr, kind=cfg.lr_schedule, warmup=cfg.warmup, n_steps=cfg.n_steps),
                        wd=cfg.weight_decay)
    gd_magic = replace(exp.magic, loader=replace(exp.loader, repeat=1))  # the same batch t as Edge Flow's substeps K t, ..., K t + K - 1
    start, n_steps = cfg.start, cfg.n_steps - cfg.start
    test_idx = exp.test_ds.subset(min(n_test, len(exp.test_ds)), index=0)
    logger = Logger((f"Edge Flow (K = {K})" if is_edgeflow else "Heun") + f" vs GD; {exp.name}", smooth=False)
    logger.log_panels = {"amplitude_along_u"}

    ef = exp.magic.train.trace(ef_state, n_steps=n_steps * K, start=start * K, pbar=False)
    gd = gd_magic.train.trace(gd_state, n_steps=n_steps, start=start, pbar=False)
    tt, s_ef = -1, None
    w0 = flat(ef_state.params).to(device)
    eigvec = None  # GD's top eigenvector, warm start
    other_eigvec = None  # Heun's
    for t, s_gd, g_gd in tqdm(gd, total=n_steps + 1, desc=f"{cfg.optimizer} vs gd"):
        while tt < t * K:
            tt, s_ef, _ = next(ef)
        assert s_ef is not None
        with torch.no_grad():
            X, Y, _ = gd_magic.loader.get_batch(min(t, start + n_steps - 1))
            w, c = flat(s_gd.params), flat(s_ef.params)
            d_wc = w - c
            logger.log(t, **{f"loss_{other}/{other}": exp.objective.forward(s_ef.params, X, Y)[0]})
            logger.log(t, **{"loss_gd/gd": gd_magic.objective.forward(s_gd.params, X, Y)[0]})
            logger.log(t, **{f"distance/gd-{other}": d_wc.norm()})
            logger.log(t, **{"travel/gd": (w - w0).norm(), f"travel/{other}": (c - w0).norm()})
            logger.log(t, **{"eig/threshold": torch.tensor(2 / s_gd.lr(t))})
            if is_edgeflow:
                u = flat(s_ef.eigvec)
                amplitude = s_ef.eps["eps"]
                proj_wc = d_wc @ u
                symmetrized = sum(exp.objective.forward({k: p + sign * amplitude * v for k, (p, v) in dict_zip(s_ef.params, s_ef.eigvec).items()}, X, Y)[0]
                                  for sign in (1, -1)) / 2
                logger.log(t, **{"loss_gd/symmetrized": symmetrized})
                logger.log(t, **{"amplitude_along_u/edgeflow": amplitude})
                if t > start:  # (0 at start: log scale)
                    logger.log(t, **{"amplitude_along_u/gd-center": proj_wc.abs()})
                logger.log(t, **{"drift_orth/gd-center": (d_wc - proj_wc * u).norm()})
                logger.log(t, **{"eig/edgeflow": s_ef.stats["sharpness"]})  # (nan at start; before GD's: drawn under it)
                if g_gd is not None:
                    g = flat(g_gd)
                    logger.log(t, **{"cos_with_u/gd_gradient": (g @ u).abs() / g.norm()})
            if eval_every and ((t - start) % eval_every == 0 or g_gd is None):
                for name, params in (("gd", s_gd.params), (other, s_ef.params)):
                    logger.log(t, **{f"test_loss/{name}": exp.objective.eval_loss(params, exp.test_loader, test_idx)[0]})
        if g_gd is not None and (t - start) % eig_every == 0:
            hvp_fn = gd_magic.objective.hvp_fn(s_gd.params, X, Y)
            if eigvec is None:
                eigvec = {k: v.detach().clone() for k, v in s_ef.eigvec.items()} if is_edgeflow else random_like(s_gd.params)
            eigval, eigvec, _ = hessian_eigs(hvp_fn, eigvec, batched=False)
            logger.log(t, **{"eig/gd": eigval})
            v = flat(eigvec) / flat(eigvec).norm()
            if is_edgeflow:
                logger.log(t, **{"cos_with_u/gd_top_eigvec": (v @ flat(s_ef.eigvec)).abs()})
            else:
                other_eigvec = random_like(s_ef.params) if other_eigvec is None else other_eigvec
                other_eigval, other_eigvec, _ = hessian_eigs(exp.objective.hvp_fn(s_ef.params, X, Y), other_eigvec, batched=False)
                logger.log(t, **{"eig/heun": other_eigval})
                g = flat(g_gd)
                logger.log(t, **{"cos_with_gd_eigvec/gd_gradient": (g @ v).abs() / g.norm()})
                if t > start:  # (w = c at start)
                    logger.log(t, **{"cos_with_gd_eigvec/gd-heun": (d_wc @ v).abs() / d_wc.norm()})

    out_dir = exp_dir / f"{cfg.optimizer}_vs_gd"
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
