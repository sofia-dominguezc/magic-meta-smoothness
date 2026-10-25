"""
Rotation of the Hessian seen from the end of training, and MAGIC's df/ds_t against it.
  python -m experiments.rayleigh DIR [--num-top-eigs 1] [--num-neg-eigs 1] [--eig-every 10] [--n-test 10] [--n-track 10]
                                     [--passes 2] [--device cpu] [--no-show]
Takes the eigenvectors v_i of sqrt(P) H sqrt(P) at the final state (DIR/full_model/final.pt, batch of step T), goes backward
through the training with `Magic.backprop` (f = loss on `n_test` test examples) and logs at each step t, to
DIR/rayleigh/metrics.pt + .png (raw lines):
- "eig/i: v^T H_t v": the Rayleigh quotient of v_i in sqrt(P_t) H_t sqrt(P_t) (batch and state of step t), and every
  `eig_every` steps "eig/i": its actual eigenvalue (warm-started LOBPCG). Equal while v_i is still an eigenvector.
- "delta_eig/i": |cos(df/ds_t, v_i)| (at T: the starting vector df/ds_T)
- "beta/<example>": df/dw_t,i / ||df/ds_t|| of `n_track` random training examples, at the steps they are in the batch
- the norms of df/ds_t (metrics_norms.png)
"""

from pathlib import Path

import click
import torch
from torch import Tensor

from main import Experiment
from src.logging import Logger
from src.lobpcg import hessian_eigs
from src.utils import dict_zip
from src.optimizers import OptimState


def rayleigh(exp_dir: Path, device: str = "cpu", num_top_eigs: int = 1, num_neg_eigs: int = 1, eig_every: int = 10,
             n_test: int = 10, n_track: int = 10, passes: int = 2):
    exp = Experiment(exp_dir, device)
    cfg, magic = exp.cfg, exp.magic
    start, T = cfg.start, cfg.n_steps
    ks = {sign: k for sign, k in ((1, num_top_eigs), (-1, num_neg_eigs)) if k > 0}  # sign -> number of eigenpairs
    logger = Logger(f"final (step {T}) Hessian eigenvectors v along training: v^T H_t v vs the eigenvalues of H_t, "
                    f"|cos| with df/ds_t; {exp.name}", smooth=False)
    logger.eos = exp.init_state.stability_threshold

    def hvp_at(state: OptimState, t: int):
        """V -> H_t V: the Hessian of the batch loss of step `t` at `state`."""
        X, Y, _ = magic.loader.get_batch(t)
        return magic.hvp_fn(state.params, X, Y)

    def log_eigs(t: int, state: OptimState, hvp_fn, warm: dict[int, dict[str, Tensor]]) -> dict[int, dict[str, Tensor]]:
        """Log the eigenvalues "eig/i" of sqrt(P_t) H_t sqrt(P_t) (LOBPCG from `warm`, random for a missing sign).
        Returns the eigenvectors {sign: {name: [k, *shape]}}."""
        eigvecs = {}
        for sign, k in ks.items():
            if sign not in warm:
                g = torch.Generator().manual_seed(t)
                warm[sign] = {name: torch.randn(k, *p.shape, generator=g, dtype=p.dtype).to(p.device) for name, p in state.params.items()}
            eigvals, eigvecs[sign], _ = hessian_eigs(hvp_fn, warm[sign], state.precond(t), sign)
            logger.log(t, **{f"eig/{sign * (i + 1)}": lam for i, lam in enumerate(eigvals.unbind())})
            if t == T:
                click.echo(f"final eigenvalues, sign {sign}: {eigvals.tolist()}")
        return eigvecs

    def log_cos(t: int, d: dict[str, Tensor]):
        """Log "delta_eig/i" = |cos(d, v_i)| for the final eigenvectors v_i (plain cosine in parameter space)."""
        d_norm = torch.stack([x.pow(2).sum() for x in d.values()]).sum() ** 0.5
        for sign, vecs in V.items():
            dots = torch.stack([(d[name] * v).flatten(1).sum(1) for name, v in vecs.items()]).sum(0)  # [k]
            v_norms = torch.stack([v.flatten(1).pow(2).sum(1) for v in vecs.values()]).sum(0) ** 0.5
            logger.log(t, **{f"delta_eig/{sign * (i + 1)}": c for i, c in enumerate((dots / (d_norm * v_norms)).abs().unbind())})

    # the final eigenvectors V, and the starting vector df/ds_T
    final = exp.load_state(exp_dir / "full_model" / "final.pt").apply(lambda s: s.to(device))
    V = log_eigs(T, final, hvp_at(final, T), {})
    X_f, Y_f = (x.to(device) for x in exp.test_ds.examples(exp.test_ds.subset(n_test, index=0)))  # the eval set of f
    params = {k: p.detach().requires_grad_() for k, p in final.params.items()}
    log_cos(T, torch.autograd.grad(magic.forward(params, X_f, Y_f)[0].mean(), params))

    # backward through the training
    tracked = set(exp.train_ds.subset(n_track, index=0).tolist()) if n_track else set()
    warm = dict(V)
    for t, state, delta, d_weights in magic.backprop.trace(
        exp.start_state, n_steps=T - start, start=start, eval_sets=[(X_f, Y_f)], passes=passes
    ):
        hvp_fn, P = hvp_at(state, t), state.precond(t)
        for sign, vecs in V.items():  # Rayleigh quotients of the final eigenvectors
            U = {name: torch.as_tensor(P[name], dtype=v.dtype, device=v.device).sqrt() * v for name, v in vecs.items()}  # sqrt(P_t) v_i
            q = torch.stack([(u * hu).flatten(1).sum(1) for u, hu in dict_zip(U, hvp_fn(U)).values()]).sum(0)  # [k]
            logger.log(t, **{f"eig/{sign * (i + 1)}: v^T H_t v": qi for i, qi in enumerate(q.unbind())})
        if eig_every and (t - start) % eig_every == 0:  # the actual eigenvalues, same batch and state
            warm = log_eigs(t, state, hvp_fn, warm)
        del hvp_fn
        log_cos(t, {name: d[0] for name, d in delta.params.items()})
        logger.log_norms(t, "delta", delta.apply(lambda d: d[0]))
        positions = {j: p for p, j in enumerate(magic.loader.batch_indices(t).tolist()) if j in tracked}  # example -> position
        norm = torch.stack([d[0].pow(2).sum() for d in delta]).sum() ** 0.5  # ||df/ds_t||
        logger.log(t, **{f"beta/{j}": d_weights[0][p] / norm for j, p in positions.items()})

    out_dir = exp_dir / "rayleigh"
    out_dir.mkdir(exist_ok=True)
    logger.finish(out_dir / "metrics.pt")
    click.echo(f"saved {out_dir / 'metrics.png'}")


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option("--device", default="cpu", show_default=True)
@click.option("--num-top-eigs", type=int, default=1, show_default=True)
@click.option("--num-neg-eigs", type=int, default=1, show_default=True)
@click.option("--eig-every", type=int, default=10, show_default=True, help="recompute the actual eigenvalues every this many steps (0: never)")
@click.option("--n-test", type=int, default=10, show_default=True, help="test examples of f (magic's first eval set)")
@click.option("--n-track", type=int, default=10, show_default=True, help="random training examples whose per-step contributions are logged (0: none)")
@click.option("--passes", type=int, default=2, show_default=True, help="re-trainings of `Magic.backprop`")
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dir, device, num_top_eigs, num_neg_eigs, eig_every, n_test, n_track, passes, show):
    rayleigh(exp_dir, device, num_top_eigs, num_neg_eigs, eig_every, n_test, n_track, passes)
    if show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
