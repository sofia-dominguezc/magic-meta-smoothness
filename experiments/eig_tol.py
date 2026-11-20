"""
How good are the LOBPCG eigenvalues at a given tolerance? Train the experiment's model (DIR/full_model/start.pt, or the
init if start = 0) and every --eig-every steps (from start + eig-every) compute the top --num-top-eigs and most negative
--num-neg-eigs eigenvalues of H (the Hessian of the step's batch mean loss, like `train`'s "eig/i") at every --tol:
  python -m experiments.eig_tol DIR [--num-top-eigs 2] [--num-neg-eigs 1] [--tol 1e-3 --tol 1e-1] [--eig-every 10] [--steps n]
                                    [--device cpu] [--no-show]
Each tolerance keeps its own warm start (as in `magic.TrainLog`), from the same random block.

Outputs in DIR/eig_tol/:
- eig_tol.png: a panel per eigenvalue (1, 2, ..., -1, -2, ...) with a line per tolerance, and the LOBPCG iterations
- results.pt: {steps [T], tols, ids, eigvals [T, tols, ids], iters [T, tols, signs (top, neg: those asked)]}
Prints the median relative difference of each tolerance's eigenvalues to the tightest one's.
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from main import Experiment
from src.lobpcg import hessian_eigs
from src.logging import fig_title


def eig_tol(exp_dir: Path, device: str = "cpu", num_top_eigs: int = 2, num_neg_eigs: int = 1, tols: tuple[float, ...] = (1e-3, 1e-1),
            eig_every: int = 10, steps: int | None = None) -> dict:
    exp = Experiment(exp_dir, device)
    cfg, magic = exp.cfg, exp.magic
    start, n_steps = exp.steps
    steps = n_steps if steps is None else steps
    tols = tuple(sorted(tols))  # tightest first
    ks = {sign: k for sign, k in ((1, num_top_eigs), (-1, num_neg_eigs)) if k > 0}  # sign -> number of eigenpairs
    ids = [sign * (i + 1) for sign, k in ks.items() for i in range(k)]
    warm: dict[tuple[int, int], dict[str, Tensor]] = {}  # (tol index, sign) -> eigenvectors [ks[sign], *shape]
    out_steps, eigvals, iters = [], [], []

    for t, state, _ in magic.train.trace(exp.start_state, n_steps=steps, start=start):
        if (t - start) % eig_every != 0 or t == start:
            continue
        X, Y, _ = magic.loader.get_batch(t)
        hvp_fn = magic.objective.hvp_fn(state.params, X, Y)
        vals_t, iters_t = [], []
        for j, tol in enumerate(tols):
            vals_j, iters_j = [], []
            for sign, k in ks.items():
                if (j, sign) not in warm:  # the same random block for every tolerance
                    g = torch.Generator().manual_seed(int(sign < 0))
                    warm[j, sign] = {name: torch.randn(k, *p.shape, generator=g, dtype=p.dtype).to(p.device) for name, p in state.params.items()}
                lam, warm[j, sign], n_iter = hessian_eigs(hvp_fn, warm[j, sign], sign, rtol=tol)
                vals_j.append(lam)
                iters_j.append(n_iter)
            vals_t.append(torch.cat(vals_j))
            iters_t.append(iters_j)
        out_steps.append(t)
        eigvals.append(torch.stack(vals_t).cpu())
        iters.append(torch.tensor(iters_t))

    results = {"steps": torch.tensor(out_steps), "tols": list(tols), "ids": ids,
               "eigvals": torch.stack(eigvals), "iters": torch.stack(iters)}
    out_dir = exp_dir / "eig_tol"
    out_dir.mkdir(exist_ok=True)
    torch.save(results, out_dir / "results.pt")

    E = results["eigvals"].double()  # [T, tols, ids]
    for j, tol in enumerate(tols[1:], 1):
        rel = ((E[:, j] - E[:, 0]).abs() / E[:, 0].abs()).median(0).values
        click.echo(f"tol {tol:g} vs {tols[0]:g}: median relative difference " + ", ".join(f"eig {i}: {d:.2e}" for i, d in zip(ids, rel.tolist())))

    fig, axes = plt.subplots(1, len(ids) + 1, figsize=(3.6 * (len(ids) + 1), 2.8), squeeze=False, layout="constrained")
    fig_title(fig, f"LOBPCG eigenvalues of the Hessian by tolerance; {exp.name}")
    for c, i in enumerate(ids):
        ax = axes[0, c]
        for j, tol in enumerate(tols):
            ax.plot(out_steps, E[:, j, c], "o-", ms=2, lw=1, color=f"C{j}", label=f"tol {tol:g}")
        ax.set(title=f"eig {i}", xlabel="step")
    ax = axes[0, -1]
    for j, tol in enumerate(tols):
        for s, sign in enumerate(ks):
            ax.plot(out_steps, results["iters"][:, j, s], "-" if sign > 0 else ":", lw=1, color=f"C{j}", label=f"tol {tol:g}, {'top' if sign > 0 else 'neg'}")
    ax.set(title="LOBPCG iterations", xlabel="step")
    ax.legend(fontsize=7)
    axes[0, 0].legend(fontsize=7)
    for ax in axes.flat:
        ax.title.set_fontsize(9)
    fig.savefig(out_dir / "eig_tol.png", dpi=150)
    click.echo(f"saved {out_dir / 'eig_tol.png'}")
    return results


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option("--device", default="cpu", show_default=True)
@click.option("--num-top-eigs", type=int, default=2, show_default=True, help="top eigenvalues (0: none)")
@click.option("--num-neg-eigs", type=int, default=1, show_default=True, help="most negative eigenvalues (0: none)")
@click.option("--tol", "tols", type=float, multiple=True, default=(0.001, 0.1), show_default=True, help="LOBPCG rtol (repeatable)")
@click.option("--eig-every", type=int, default=10, show_default=True)
@click.option("--steps", type=int, default=None, help="training steps from start (default: the whole training)")
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dir, device, num_top_eigs, num_neg_eigs, tols, eig_every, steps, show):
    eig_tol(exp_dir, device, num_top_eigs, num_neg_eigs, tols, eig_every, steps)
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()


# 0.001 and 0.1 are effectively equal
# top negative eig: results\small_cnn\sgd2\eig_tol\eig_tol.png
# top2 positive eigs: results\small_cnn\sgd2\eig_tol\eig_tol_pos.png
