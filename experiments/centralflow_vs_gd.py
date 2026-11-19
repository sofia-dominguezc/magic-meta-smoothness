"""
Does the central flow track the gradient descent it models? Compare CF_DIR (`--optimizer centralflow`) with GD_DIR (`sgd`,
the same config otherwise), both trained with `train --checkpoint-every`, at the GD steps where both have a checkpoint
(a central flow checkpoint at substep K t is GD's step t, K = SUBSTEPS), and from their logged metrics:
  python -m experiments.centralflow_vs_gd CF_DIR GD_DIR [--n-test 1000] [--window 20] [--device cpu] [--no-show]

Outputs in CF_DIR/vs_gd/ (metrics.pt and metrics.png, panels in GD steps), w = GD's params, c = the central flow's:
- loss: GD's logged batch loss and its moving average over --window steps, the central flow's batch loss, and its prediction
  of GD's time-averaged loss: loss(c) + oscillation / lr (oscillation = tr Sigma, logged by the step)
- test_loss, test_acc: of w and c at the checkpoints, on --n-test test examples
- distance: |w - c|; travel: |w - w_0|, |c - w_0| (how far the models move at all)
- in_span: the fraction of |w - c|^2 in the span of the critical eigenvectors the central flow's checkpoint stores
- oscillation: the central flow's tr Sigma (GD's mean squared displacement from c, predicted) and the checkpoints' |U^T (w - c)|^2
  (one sample of it)
- eig: the top Hessian eigenvalue logged by both trainings, and 2 / lr; n_critical: the central flow's
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from main import Experiment
from src.logging import Logger
from src.second_order import CentralFlowState


def flat(d: dict[str, Tensor]) -> Tensor:
    return torch.cat([t.detach().flatten() for t in d.values()])


def centralflow_vs_gd(cf_dir: Path, gd_dir: Path, device: str = "cpu", n_test: int = 1000, window: int = 20) -> dict:
    cf, gd = Experiment(cf_dir, device), Experiment(gd_dir, device)
    if cf.cfg.optimizer != "centralflow" or gd.cfg.optimizer != "sgd":
        raise click.UsageError(f"optimizers {cf.cfg.optimizer} (CF_DIR) and {gd.cfg.optimizer} (GD_DIR): expected centralflow and sgd")
    K, lr = cf.substeps, cf.cfg.lr
    logger = Logger(f"Central flow (K = {K}) vs GD; {cf.name}", smooth=False, step_scale=K)

    # the logged curves, at substeps (GD's step t at K t)
    cf_log = torch.load(cf_dir / "full_model" / "metrics.pt", weights_only=True)
    gd_log = torch.load(gd_dir / "full_model" / "metrics.pt", weights_only=True)
    for name, series in (("gd", gd_log["loss"]), ("gd_avg", gd_log["loss"])):
        steps, values = series["step"].numpy(), series["value"].numpy()
        if name == "gd_avg" and len(values) >= window:  # centered moving average
            values = np.convolve(values, np.ones(window) / window, mode="valid")
            steps = steps[window - 1 :] - (window - 1) / 2
        for s, v in zip(steps, values):
            logger.log(round(K * float(s)), **{f"loss/{name}": torch.tensor(float(v))})
    for s, v in zip(cf_log["loss"]["step"], cf_log["loss"]["value"]):
        logger.log(int(s), **{"loss/cf": v})
    for s, v, osc in zip(cf_log["loss"]["step"], cf_log["loss"]["value"], cf_log["oscillation"]["value"]):
        logger.log(int(s), **{"loss/cf+osc": v + osc / lr})
    for s, v in zip(cf_log["oscillation"]["step"], cf_log["oscillation"]["value"]):
        logger.log(int(s), **{"oscillation/cf": v})
    for s, v in zip(cf_log["n_critical"]["step"], cf_log["n_critical"]["value"]):
        logger.log(int(s), **{"n_critical/cf": v})
    for s, v in zip(cf_log["eig/1"]["step"], cf_log["eig/1"]["value"]):
        logger.log(int(s), **{"eig/cf": v})
    if "eig/1" in gd_log:
        for s, v in zip(gd_log["eig/1"]["step"], gd_log["eig/1"]["value"]):
            logger.log(int(K * s), **{"eig/gd": v})
    for s in (0, cf.steps[1]):
        logger.log(s, **{"eig/threshold": torch.tensor(2 / lr)})

    # the checkpoints
    cf_ckpts = {t // K: path for t, path in cf.checkpoints().items() if t % K == 0}
    gd_ckpts = gd.checkpoints()
    common = sorted(set(cf_ckpts) & set(gd_ckpts))
    if not common:
        raise click.UsageError(f"no common checkpoint step: central flow has {sorted(cf_ckpts)}, GD has {sorted(gd_ckpts)}")
    test_idx = torch.randperm(len(cf.test_ds), generator=torch.Generator().manual_seed(0))[:n_test]
    w0 = flat(cf.start_state.params).to(device)
    for t in common:
        c_state, w_state = cf.load_state(cf_ckpts[t]), gd.load_state(gd_ckpts[t])
        assert isinstance(c_state, CentralFlowState)
        c_params = {k: p.to(device) for k, p in c_state.params.items()}
        w_params = {k: p.to(device) for k, p in w_state.params.items()}
        with torch.no_grad():
            c, w = flat(c_params), flat(w_params)
            d = w - c
            logger.log(K * t, **{"distance/gd-cf": d.norm(), "travel/gd": (w - w0).norm(), "travel/cf": (c - w0).norm()})
            n_critical = int(c_state.stats["n_critical"])
            if n_critical:
                U = torch.cat([u[:n_critical].flatten(1) for u in c_state.eigvecs.values()], 1).to(device)  # [k, N]
                proj = (U @ d).square().sum()
                logger.log(K * t, **{"in_span/gd-cf": proj / d.square().sum(), "oscillation/gd": proj})
            for name, params in (("gd", w_params), ("cf", c_params)):
                loss, metrics = cf.objective.eval_loss(params, cf.test_loader, test_idx)
                logger.log(K * t, **{f"test_loss/{name}": loss, f"test_acc/{name}": metrics["acc"]})
    out_dir = cf_dir / "vs_gd"
    out_dir.mkdir(exist_ok=True)
    logger.finish(out_dir / "metrics.pt")
    return logger.metrics()


@click.command(help=__doc__)
@click.argument("cf_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("gd_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--n-test", type=int, default=1000, show_default=True, help="test examples (a fixed random subset)")
@click.option("--window", type=int, default=20, show_default=True, help="GD steps of the moving average of GD's loss")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--show/--no-show", default=True, help="show the plots")
def main(cf_dir, gd_dir, n_test, window, device, show):
    centralflow_vs_gd(cf_dir, gd_dir, device, n_test, window)
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
