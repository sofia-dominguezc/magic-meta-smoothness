"""

Train the experiment's full model for the detailed logs (DIR/logged/metrics.pt + plots; the model is saved to
DIR/full_model/final.pt if it isn't there yet):
  python train_logged.py DIR [--num-top-eigs 1] [--num-neg-eigs 0] [--eig-every 10] [--no-tangent] [--eval-every 100]
                             [--n-test 1000] [--ema-decay 0.9] [--device cpu] [--no-show]

On top of `Magic.train`'s own logs, from the states and gradients `Magic.train.trace` yields (with `create_graph`: not compiled):
- "eig/i": top (i > 0) / most negative (i < 0) eigenvalues of the preconditioned Hessian of the batch loss, "eigvec/i" their norms.
  The eigenpair a 2nd-order step computes (sharpness: the top, newton: the most negative) is logged at every step with the
  step size it set (sharpness: 2 lr by construction); LOBPCG only runs for 2 or more eigenpairs of that sign
- "tangent": the norms of ds_t/ds_0 v for a random v (SGD only: the step's Jacobian is symmetric, so backward AD through the
  step propagates it); "tangent_eig/i": its |cos| with the eigenvectors
- "<name>/ema": with an EMA of the params; "ema_eig/i": |cos| of params - EMA with the eigenvectors
- "test_<name>": on random test examples every `eval_every` steps (the whole test set at the last step)
- "lr" if scheduled
"""

from dataclasses import asdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from main import Experiment
from src.data import Loader
from src.logging import Logger, op_norm
from src.lobpcg import hessian_eigs
from src.magic import Magic
from src.utils import dict_zip, hvp_from_grads, metrics
from src.optimizers import NewtonState, OptimState, SGDState, SharpnessState


def train_logged(
    magic: Magic, init_state: OptimState, *, n_steps: int, start: int = 0, test_loader: Loader, logger: Logger,
    n_top_eigs: int = 1, n_neg_eigs: int = 0, eig_every: int = 10, tangent: bool = True,
    eval_every: int = 100, n_test: int | None = None, ema_decay: float | None = 0.9,
) -> OptimState:
    """Returns the trained state."""
    second_order = {1: "sharpness"} if isinstance(init_state, SharpnessState) \
        else {-1: "min_eig"} if isinstance(init_state, NewtonState) \
        else {}
    ks = {sign: k for sign, k in ((1, n_top_eigs), (-1, n_neg_eigs)) if k > 1 or k == 1 and sign not in second_order}  # sign -> eigenpairs by LOBPCG
    stored = {sign: name for sign, name in second_order.items() if sign not in ks}  # sign -> the stat of the eigenpair the step keeps (aux `eigvec`)
    if tangent and type(init_state) is not SGDState:
        click.echo("no tangent: only an SGD step has a symmetric Jacobian")
        tangent = False
    logger.eos = init_state.stability_threshold
    warm: dict[int, dict[str, Tensor]] = {}  # sign -> eigenvectors [k, *shape] per param, LOBPCG's warm start
    ema: dict[str, Tensor] | None = None
    v: OptimState | None = None
    prev_directions: dict[str, dict[str, Tensor] | None] = {}  # of the previous step, for the stored eigenpair
    if tangent:
        g = torch.Generator().manual_seed(start)
        v = init_state.apply(lambda s: torch.randn(s.shape, generator=g, dtype=s.dtype).to(magic.device))
        v = v.apply(lambda s: s / op_norm(s, n_iter=10)[0])

    def cos(d: dict[str, Tensor], eigvecs: dict[str, Tensor]) -> Tensor:
        """|cos| of the direction `d` with each of the k eigenvectors -> [k]"""
        dots = torch.stack([(d[name] * vec).flatten(1).sum(1) for name, vec in eigvecs.items()]).sum(0)
        d_norm = torch.stack([x.pow(2).sum() for x in d.values()]).sum() ** 0.5
        v_norms = torch.stack([vec.flatten(1).pow(2).sum(1) for vec in eigvecs.values()]).sum(0) ** 0.5
        return (dots / (d_norm * v_norms)).abs()

    def log_eigs(t: int, ids: list[int], eigvals: Tensor, eigvecs: dict[str, Tensor], directions: dict[str, dict[str, Tensor] | None]):
        """Log "eig/i", the norms "eigvec/i" and the |cos| with `directions` of the eigenpairs `ids` (eigvecs: [k, *shape] per param)"""
        logger.log(t, **{f"eig/{i}": lam for i, lam in zip(ids, eigvals.unbind())})
        for j, i in enumerate(ids):
            logger.log_norms(t, f"eigvec/{i}", {name: vec[j] for name, vec in eigvecs.items()})
        for name, d in directions.items():
            if d is not None:
                logger.log(t, **{f"{name}/{i}": c for i, c in zip(ids, cos(d, eigvecs).unbind())})

    for t, state, grads in magic.train.trace(init_state, n_steps=n_steps, start=start, log=logger, create_graph=bool(ks) or tangent):
        params = {k: p.detach() for k, p in state.params.items()}
        X, Y, _ = magic.loader.get_batch(t)
        if state.lr.kind != "constant" or state.lr.warmup:
            logger.log(t, lr=torch.tensor(state.lr(t)))

        if ema_decay is not None:
            with torch.no_grad():
                if ema is None:
                    ema = {k: p.clone() for k, p in params.items()}
                else:
                    ema = {k: e + (1 - ema_decay) * (p - e) for k, (e, p) in dict_zip(ema, params).items()}
                losses, out = magic.forward(ema, X, Y)
                batch_metrics = {"loss": losses.mean(), **{k: m.mean() for k, m in metrics(out, Y).items()}}
                logger.log(t, **{f"{k}/ema": m for k, m in batch_metrics.items()})

        if v is not None:
            logger.log_norms(t, "tangent", v)

        directions = {"tangent_eig": None if v is None else v.params,
                      "ema_eig": None if ema is None or t == start else {k: p - e for k, (p, e) in dict_zip(params, ema).items()}}
        if stored and t > start:  # the eigenpair step t - 1 computed at s_{t-1}, with the (scalar) step size P it set: P eig
            P = next(iter(state.precond(t).values()))
            for sign, name in stored.items():
                log_eigs(t - 1, [sign], state.stats[name][None] * P, {k: u[None] for k, u in state.eigvec.items()}, prev_directions)
        if ks and grads is not None and (t - start) % eig_every == 0 and (t > start or not second_order):  # (2nd order: no P at start)
            hvp_fn, P = hvp_from_grads(grads, state.params), state.precond(t)
            for sign, k in ks.items():
                if sign not in warm:  # random start
                    g = torch.Generator().manual_seed(t)
                    warm[sign] = {name: torch.randn(k, *p.shape, generator=g, dtype=p.dtype).to(p.device) for name, p in params.items()}
                eigvals, warm[sign], n_iter = hessian_eigs(hvp_fn, warm[sign], P, sign)
                logger.log(t, **{f"lobpcg_iters/{sign}": torch.tensor(n_iter)})
                log_eigs(t, [sign * (i + 1) for i in range(k)], eigvals, warm[sign], directions)
        prev_directions = directions

        if v is not None and grads is not None:  # v <- (ds_{t+1}/ds_t)^T v
            v = state.unflatten(torch.autograd.grad(list(state.step(t, grads)), list(state), list(v)))

        if eval_every and ((t - start) % eval_every == 0 or grads is None):
            test_ds = test_loader.dataset
            indices = torch.arange(len(test_ds)) if n_test is None or grads is None else test_ds.subset(n_test, index=t)
            for p, suffix in ((params, ""), (ema, "/ema")):
                if p is not None:
                    loss, mean_metrics = magic.eval_loss(p, test_loader, indices)
                    logger.log(t, **{f"test_{k}{suffix}": m for k, m in {"loss": loss, **mean_metrics}.items()})
    return state.apply(lambda s: s.detach().cpu())


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--compile/--no-compile", default=False, help="torch.compile the model forward (only without eigenvalues and tangent)")
@click.option("--num-top-eigs", type=int, default=1, show_default=True, help="log the top Hessian eigenvalues (0: don't; sharpness: the step's own for 0 or 1)")
@click.option("--num-neg-eigs", type=int, default=0, show_default=True, help="log the most negative Hessian eigenvalues (0: don't; newton: the step's own for 0 or 1)")
@click.option("--eig-every", type=int, default=10, show_default=True, help="log the eigenvalues every this many steps")
@click.option("--tangent/--no-tangent", default=True, help="log the norms of ds_t/ds_0 v for a random v (SGD only)")
@click.option("--eval-every", type=int, default=100, show_default=True, help="log the test loss and metrics every this many steps (0: never)")
@click.option("--n-test", type=int, default=1000, show_default=True, help="test examples (a fresh random subset at each eval, the whole test set at the last step) the test loss is logged on (0: always the whole test set)")
@click.option("--ema-decay", type=float, default=0.9, show_default=True, help="also log the loss and metrics of an EMA of the params with this decay, and its |cos| with the Hessian eigenvectors (0: don't)")
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dir, device, compile, num_top_eigs, num_neg_eigs, eig_every, tangent, eval_every, n_test, ema_decay, show):
    exp = Experiment(exp_dir, device, compile)
    cfg = exp.cfg
    logger = Logger(exp.name)
    final = train_logged(
        exp.magic, exp.start_state, n_steps=cfg.n_steps - cfg.start, start=cfg.start, test_loader=exp.test_loader, logger=logger,
        n_top_eigs=num_top_eigs, n_neg_eigs=num_neg_eigs, eig_every=eig_every, tangent=tangent,
        eval_every=eval_every, n_test=n_test or None, ema_decay=ema_decay or None,
    )
    if not (exp_dir / "full_model" / "final.pt").exists():
        (exp_dir / "full_model").mkdir(exist_ok=True)
        torch.save(asdict(final), exp_dir / "full_model" / "final.pt")
    out_dir = exp_dir / "logged"
    out_dir.mkdir(exist_ok=True)
    logger.finish(out_dir / "metrics.pt")
    click.echo(f"saved {out_dir / 'metrics.png'}")
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
