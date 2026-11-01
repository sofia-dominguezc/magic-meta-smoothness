"""
MAGIC with the Hessian and Gauss-Newton eigenvalues along the backward: how large the top eigenvalues of the batch Hessian
(and Gauss-Newton matrix) at s_t are, and how much df/ds_t aligns with their eigenvectors.
  python -m experiments.magic_eigs DIR [--n-test 10] [--n-subsets 1] [--n-eigs 3] [--n-neg-eigs 0] [--eig-every 10] [--variant hessian] [--device cpu]

Runs `Magic.backprop` (the steps `start`-`n_steps` of DIR's config, like `main.py magic --variant`) and, every `eig_every` steps,
LOBPCG on the Hessian of the batch loss at the state s_t it yields (second derivatives from the params; warm-started) and on
the Gauss-Newton matrix (`Magic.gnvp_fn`). Saves DIR/magic_eigs[_<variant>]/metrics.pt and its plots:
- "eig/i": the top (i > 0) / most negative (i < 0) Hessian eigenvalues, "eig/gn<i>": the top Gauss-Newton eigenvalues,
  and "eig/threshold" = stability threshold / step size (scalar step sizes)
- "delta_eig/i", "delta_eig/gn<i>": |cos| of df/dparams_t with the eigenvectors, one line per eval set
- "delta/<group>/<set>": the total L2 norms of df/ds_t and "displacement/delta/<group>/<set>" their relative change per step,
  "d_weights/<set>": |df/dw_t|
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from main import HESSIAN, Experiment, eig_cos, parse_variant, variant_option
from src.lobpcg import hessian_eigs
from src.logging import Logger
from src.utils import HVP


def magic_eigs(
    exp_dir: Path, *, n_test: int = 1, n_subsets: int = 1, n_eigs: int = 3, n_neg_eigs: int = 0, eig_every: int = 10,
    passes: int = 2, variant: str = HESSIAN, device: str = "cpu", compile: bool = False, show: bool = True,
) -> Path:
    exp = Experiment(exp_dir, device, compile)
    cfg, magic = exp.cfg, exp.magic
    gn_below, skip_below = (None if s is None else s * exp.substeps for s in parse_variant(variant, cfg))
    out_dir = exp_dir / ("magic_eigs" if variant == HESSIAN else f"magic_eigs_{variant}")
    eval_sets = [exp.test_ds.examples(exp.test_ds.subset(n_test, index=i)) for i in range(n_subsets)]
    ks = {sign: k for sign, k in ((1, n_eigs), (-1, n_neg_eigs)) if k}  # Hessian: sign -> eigenpairs; the GN matrix: the top n_eigs
    logger = Logger(f"MAGIC{'' if variant == HESSIAN else f' ({variant})'} vs the Hessian eigenvalues (every {eig_every} steps); {exp.name}",
                    smooth=False, total_norms=True)
    eos = exp.start_state.stability_threshold  # logged normalized by the step size as "eig/threshold"
    warm: dict[int | str, dict[str, Tensor]] = {}  # sign (or "gn") -> eigenvectors [k, *shape] per param, LOBPCG's warm start

    start, n_steps = exp.steps
    trace = magic.backprop.trace(
        exp.start_state, n_steps=n_steps, start=start, eval_sets=eval_sets, passes=passes,
        gn_below=gn_below, skip_below=skip_below, log=logger,
    )
    for t, state, delta, _ in trace:
        if (t - start) % eig_every:
            continue
        X, Y, _ = magic.loader.get_batch(t)
        params = {k: p.detach() for k, p in state.params.items()}
        hvp_fn = magic.hvp_fn(params, X, Y)  # the 2nd-derivative graph of the batch loss at s_t
        d = {k: v.detach() for k, v in delta.params.items()}  # [n_fns, *shape]
        matrices: list[tuple[HVP, int | str, int, list[str]]] = [(hvp_fn, sign, k, [str(sign * (i + 1)) for i in range(k)]) for sign, k in ks.items()]
        if n_eigs:
            matrices.append((magic.gnvp_fn(params, X, Y), "gn", n_eigs, [f"gn{i + 1}" for i in range(n_eigs)]))
        for mvp, key, k, ids in matrices:
            if key not in warm:
                g = torch.Generator().manual_seed(t)
                warm[key] = {name: torch.randn(k, *p.shape, generator=g, dtype=p.dtype).to(p.device) for name, p in params.items()}
            eigvals, warm[key], n_iter = hessian_eigs(mvp, warm[key], None, -1 if key == -1 else 1)
            logger.log(t, **{f"eig/{i}": lam for i, lam in zip(ids, eigvals.unbind())}, **{f"lobpcg_iters/{key}": torch.tensor(n_iter)})
            logger.log(t, **{f"delta_eig/{i}": c for i, c in zip(ids, eig_cos(d, warm[key]).unbind(1))})  # [n_fns] each: a line per eval set
        P = next(iter(state.precond(t).values()))
        if not isinstance(P, Tensor) or P.ndim == 0:
            logger.log(t, **{"eig/threshold": torch.as_tensor(eos / P, dtype=torch.get_default_dtype())})
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.finish(out_dir / "metrics.pt")
    click.echo(f"saved {out_dir / 'metrics.pt'} and its plots")
    if show:
        plt.show()
    plt.close("all")
    return out_dir


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--n-test", type=int, default=1, show_default=True, help="test examples in each eval set")
@click.option("--n-subsets", type=int, default=1, show_default=True, help="eval sets (a line each in the |cos| panels)")
@click.option("--n-eigs", type=int, default=3, show_default=True, help="top eigenvalues")
@click.option("--n-neg-eigs", type=int, default=0, show_default=True, help="most negative eigenvalues")
@click.option("--eig-every", type=int, default=10, show_default=True, help="steps between eigenvalue computations")
@click.option("--passes", type=int, default=2, show_default=True, help="re-trainings of the replay")
@variant_option
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--compile/--no-compile", default=False, help="torch.compile the model forward of the replays")
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dir, n_test, n_subsets, n_eigs, n_neg_eigs, eig_every, passes, variant, device, compile, show):
    magic_eigs(exp_dir, n_test=n_test, n_subsets=n_subsets, n_eigs=n_eigs, n_neg_eigs=n_neg_eigs, eig_every=eig_every, passes=passes,
               variant=variant, device=device, compile=compile, show=show)


if __name__ == "__main__":
    main()
