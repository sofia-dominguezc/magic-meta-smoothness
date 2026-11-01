"""

Is training chaotic or structured at an unstable point? Train copies of the experiment's model (DIR/full_model/start.pt,
or the init if start = 0) that differ by a small perturbation, vmapped together, and track how they drift apart:
  python -m experiments.drift DIR [--n-models 8] [--eps 1e-3] [--steps n] [--direction randn] [--device cpu] [--no-show]

Models (rows of the vmapped training): "center" (unperturbed), "1" ... "n" (center + eps * direction_i, every tensor of
direction_i has operator norm 1, so each tensor moves by eps in the norm of the norm plots; --direction: how the
directions are sampled, `DIRECTIONS`) and "ref" (the center trained plain, unvmapped: its difference to "center" is the
fp32 rounding of the two kernel paths, the noise floor the perturbations are compared to). Only the parameters are
perturbed, not the optimizer buffers. Distances between models are the max over tensors of the operator norm of the
difference (= eps at the start). The two trainings run side by side (two `Magic.train.trace` zipped), everything is
computed on the live states: nothing is checkpointed.

Outputs in DIR/drift/<direction>_eps<eps>_n<n>_steps<steps>/ (--replot redoes the plots from results.json):
- drift.png: (1) distance of every model to the center at every step (log scale), with ref; (2) their losses on
  --n-test fixed test examples; (3) trajectories of the displacements from the center (kept every --traj-every steps),
  projected on the 2 principal directions of the final displacements (a fan = structure along few directions; a cloud = chaos)
- final.png, for the final models (rows/columns ordered by hierarchical clustering of the distances, center first,
  ref last; the same colors as drift.png): pairwise distances; pairwise cosines of the displacements from the center;
  pairwise distances in function space (RMS of the per-example loss differences on the test examples); per-example
  loss - center's loss (examples sorted by their spread across models); cosines of the initial directions and of the
  final displacements with the top / most negative --n-eigs eigenvectors of the (preconditioned) Hessian at the start
  state (on the first training batch, as logged by `train`); singular values of the final displacements
  (participation ratio = effective number of directions the models spread along)
- results.json: all of the above as numbers, plus summary statistics
"""

import json
from pathlib import Path
from typing import Callable

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from torch import Tensor

from main import Experiment
from src.lobpcg import hessian_eigs
from src.logging import fig_title
from src.utils import dict_zip


def spec_norms(D: Tensor) -> Tensor:
    """Exact operator norms (the norm of `logging.op_norm`) of a batch [R, *shape] of parameter-shaped tensors -> [R]:
    L2 for vectors, spectral of the [out, -1] matrices."""
    return D.norm(dim=-1) if D.ndim <= 2 else torch.linalg.matrix_norm(D.flatten(2), ord=2)


def randn_direction(params: dict[str, Tensor], index: int) -> dict[str, Tensor]:
    """Gaussian, each tensor normalized to operator norm 1 (like the tangent of main.train_logged)."""
    g = torch.Generator().manual_seed(index)
    d = {k: torch.randn(p.shape, generator=g, dtype=p.dtype) for k, p in params.items()}
    return {k: t / spec_norms(t[None])[0] for k, t in d.items()}


DIRECTIONS: dict[str, Callable[[dict[str, Tensor], int], dict[str, Tensor]]] = {"randn": randn_direction}


def flat(params: dict[str, Tensor]) -> Tensor:
    """[R, *shape] per tensor -> [R, P]"""
    return torch.cat([p.flatten(1) for p in params.values()], 1)


def run(exp: Experiment, out_dir: Path, n_models: int, eps: float, steps: int, direction: str, traj_every: int,
        n_test: int, n_eigs: int) -> dict:
    """Train center + n perturbed models (vmapped) next to the plain reference, analyze along the way; everything in results.json."""
    cfg, magic, n = exp.cfg, exp.magic, n_models
    start_state = exp.start_state.apply(lambda t: t.cpu())
    directions = [DIRECTIONS[direction](start_state.params, i) for i in range(1, n + 1)]
    batched = start_state.apply(lambda t: t.expand(n + 1, *t.shape).clone())
    for i, d in enumerate(directions, 1):
        for k, t in d.items():
            batched.params[k][i] += eps * t
    rows = ["center", *map(str, range(1, n + 1)), "ref"]
    test_idx = exp.test_ds.subset(n_test)
    X, Y = (x.to(exp.device) for x in exp.test_ds.examples(test_idx))
    example_losses = torch.func.vmap(lambda p: exp.magic.forward(p, X, Y)[0])  # batched params -> [R, n_test]

    click.echo(f"training the reference and {n + 1} models (vmapped) for {steps} steps from step {cfg.start}")
    dists, losses, traj_steps, disps = [], [], [], []  # per step: [n + 1] (ref last), [n + 2]; every traj_every steps: [n + 1, P]
    vmapped = magic.train.trace(batched, n_steps=steps, start=cfg.start)
    plain = magic.train.trace(start_state, n_steps=steps, start=cfg.start, pbar=False)
    for (t, s, _), (_, ref, _) in zip(vmapped, plain):  # (the yielded states are detached: no graphs here)
        params = {k: torch.cat([p, r[None]]) for k, (p, r) in dict_zip(s.params, ref.params).items()}  # [n + 2, ...]: center, 1..n, ref
        D = {k: (p[1:] - p[:1]).double() for k, p in params.items()}  # displacements from the center [n + 1, ...]
        dists.append(torch.stack([spec_norms(d) for d in D.values()]).max(0).values)
        ell = example_losses(params).double()  # [n + 2, n_test]
        losses.append(ell.mean(1))
        if (t - cfg.start) % traj_every == 0 or t == cfg.start + steps:
            traj_steps.append(t)
            disps.append(flat(D).float().cpu())
    dists, losses = torch.stack(dists).cpu().numpy(), torch.stack(losses).cpu().numpy()  # [steps + 1, n + 1], [steps + 1, n + 2]
    all_steps = list(range(cfg.start, cfg.start + steps + 1))

    # the final models: trajectories on the principal directions of the final displacements, pairwise distances in
    # parameter and function space, cosines of the displacements
    D_final = disps[-1].double()  # [n + 1, P], ref last
    disp, Dn = D_final[:n], D_final / D_final.norm(dim=1, keepdim=True)
    _, svals, Vh = torch.linalg.svd(disp, full_matrices=False)
    traj = torch.stack([d.double() @ Vh[:2].T for d in disps]).numpy()  # [n_traj, n + 1, 2]
    pdist = torch.stack([spec_norms((p[:, None] - p[None]).flatten(0, 1).double()).view(n + 2, n + 2) for p in params.values()]).max(0).values.cpu().numpy()
    ell = ell.cpu().numpy()
    fdist = np.sqrt(((ell[:, None] - ell[None]) ** 2).mean(-1))
    cos = (Dn @ Dn.T).numpy()
    clustered = leaves_list(linkage(squareform(pdist[1:-1, 1:-1], checks=False), "average", optimal_ordering=True)) if n > 1 else [0]
    order = [0, *(1 + np.array(clustered)).tolist(), n + 1]  # center, perturbed in clustered order, ref

    # the directions vs. the Hessian at the start state
    X0, Y0, _ = magic.loader.get_batch(cfg.start)
    params0 = {k: p.to(magic.device) for k, p in start_state.params.items()}
    hvp_fn = exp.magic.hvp_fn(params0, X0, Y0)  # V -> H V on batched params dicts
    eig_names, eigvecs, eigvals, g = [], [], [], torch.Generator().manual_seed(0)
    for sign in [1, -1]:
        start = {k: torch.randn(n_eigs, *p.shape, generator=g, dtype=p.dtype).to(p.device) for k, p in params0.items()}
        vals, vecs, _ = hessian_eigs(hvp_fn, start, start_state.precond(cfg.start), sign)
        eig_names += [f"eig {'-' if sign < 0 else ''}{i + 1}" for i in range(n_eigs)]
        eigvecs.append(flat(vecs).T.double().cpu())  # [P, n_eigs]
        eigvals += (vals.detach().cpu() * (2 / start_state.stability_threshold)).tolist()
    V = torch.cat(eigvecs, dim=1)  # [P, 2 n_eigs]
    dirs = torch.cat([flat({k: t[None] for k, t in d.items()}) for d in directions]).double()  # [n, P]
    eig_cos = {
        "initial direction": ((dirs / dirs.norm(dim=1, keepdim=True)) @ V).numpy(),  # [n, 2 n_eigs]
        "final displacement": (Dn @ V).numpy(),  # [n + 1, 2 n_eigs]
    }

    results = {
        "eps": eps, "steps": all_steps, "traj_steps": traj_steps, "rows": rows, "order": order,
        "dist_to_center": dists.tolist(), "test_loss": losses.tolist(), "trajectories": traj.tolist(),
        "pairwise_dist": pdist.tolist(), "pairwise_function_dist": fdist.tolist(), "displacement_cos": cos.tolist(),
        "example_loss": ell.tolist(), "eigvals": dict(zip(eig_names, eigvals)),
        "eig_cos": {k: v.tolist() for k, v in eig_cos.items()}, "singular_values": svals.tolist(),
        "summary": {
            "final dist to center: ref": float(dists[-1, -1]), "perturbed mean": float(dists[-1, :-1].mean()),
            "perturbed min": float(dists[-1, :-1].min()), "perturbed max": float(dists[-1, :-1].max()),
            "pairwise dist of perturbed: mean": float(pdist[1:-1, 1:-1][np.triu_indices(n, 1)].mean()),
            "participation ratio": float(svals.sum() ** 2 / (svals**2).sum()) if n > 1 else 1.0,  # of the squared singular values
            "test loss: center": float(losses[-1, 0]), "ref": float(losses[-1, -1]),
            "perturbed std": float(losses[-1, 1:-1].std()),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    for k, v in results["summary"].items():
        click.echo(f"  {k}: {v:.4g}")
    return results


def plot(out_dir: Path, results: dict, title: str, show: bool):
    steps, rows, order = np.array(results["steps"]), results["rows"], results["order"]
    n = len(rows) - 2
    colors = {i: plt.cm.viridis(j / max(1, n - 1)) for j, i in enumerate(order[1:-1])}  # perturbed models, in clustered order
    colors[0], colors[n + 1] = "red", "black"
    styles = {0: dict(ls="-", lw=1.5), n + 1: dict(ls="--", lw=1.2)}
    title = f"{n} models perturbed by eps = {results['eps']:g} from step {steps[0]}, {steps[-1] - steps[0]} steps; {title}"

    dists, losses, traj = (np.array(results[k]) for k in ["dist_to_center", "test_loss", "trajectories"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    for i in range(1, n + 2):  # perturbed + ref (last column of dists)
        axes[0].plot(steps, np.where(dists[:, i - 1] > 0, dists[:, i - 1], np.nan), color=colors[i], label=rows[i], **styles.get(i, dict(lw=1)))
    axes[0].axhline(results["eps"], color="gray", ls=":", lw=0.8, label="eps")
    axes[0].set(title="distance to the center (max over tensors of the op norm)", xlabel="step", yscale="log")
    axes[0].legend(fontsize=7, ncol=2)
    for i in range(n + 2):
        axes[1].plot(steps, losses[:, i], color=colors[i], label=rows[i], **styles.get(i, dict(lw=1)))
    axes[1].set(title=f"test loss ({len(results['example_loss'][0])} fixed examples)", xlabel="step")
    axes[1].legend(fontsize=7, ncol=2)
    for i in range(1, n + 2):
        axes[2].plot(traj[:, i - 1, 0], traj[:, i - 1, 1], color=colors[i], label=rows[i], **styles.get(i, dict(lw=1)))
        axes[2].plot(*traj[-1, i - 1], "o", ms=4, color=colors[i])
    axes[2].plot(0, 0, "x", color="red", ms=8, label="center")
    axes[2].set(title="displacements from the center along training\n(on the 2 principal directions of the final ones)",
                xlabel="PC 1", ylabel="PC 2", aspect="equal")
    axes[2].legend(fontsize=7, ncol=2)
    fig_title(fig, title)
    fig.savefig(out_dir / "drift.png", dpi=150)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), layout="constrained")
    labels = [rows[i] for i in order]
    ell = np.array(results["example_loss"])
    ell_diff = (ell - ell[0])[order]
    ex_order = np.argsort(-ell[1:-1].std(0))[:200]  # the examples the models disagree on most
    init_cos, final_cos = (np.array(results["eig_cos"][k]) for k in ["initial direction", "final displacement"])  # [n, 2k], [n + 1, 2k]
    init_cos = np.concatenate([init_cos, np.full((1, init_cos.shape[1]), np.nan)])  # ref has no initial direction
    disp_order = [i - 1 for i in order[1:]]  # displacement i is of model i + 1
    eig_mat = np.concatenate([init_cos, final_cos], axis=1)[disp_order]
    eig_labels = [f"{k}\n{v:.2f}" for k, v in results["eigvals"].items()]  # with the normalized eigenvalue (EoS at 2)
    matrices = [  # (ax, title, matrix, xticklabels, cmap, vmin, vmax)
        (axes[0, 0], "pairwise distance", np.array(results["pairwise_dist"])[np.ix_(order, order)], labels, "viridis", 0, None),
        (axes[0, 1], "pairwise cosine of the displacements from the center", np.array(results["displacement_cos"])[np.ix_(disp_order, disp_order)], labels[1:], "RdBu", -1, 1),
        (axes[0, 2], "pairwise distance in function space (RMS loss difference)", np.array(results["pairwise_function_dist"])[np.ix_(order, order)], labels, "viridis", 0, None),
        (axes[1, 0], "per-example loss - center's (examples sorted by spread)", ell_diff[:, ex_order], None, "RdBu", -np.abs(ell_diff).max(), np.abs(ell_diff).max()),
        (axes[1, 1], "cosine with the Hessian eigenvectors at the start:\ninitial direction | final displacement",
         eig_mat, eig_labels * 2, "RdBu", -1, 1),
    ]
    for ax, name, M, xticks, cmap, vmin, vmax in matrices:
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set(title=name, yticks=range(M.shape[0]), yticklabels=labels if M.shape[0] == len(labels) else labels[1:])
        if xticks is not None:
            ax.set(xticks=range(M.shape[1]), xticklabels=xticks)
            ax.tick_params(axis="x", labelsize=7)
        else:
            ax.set(xlabel="test example", xticks=[])
        ax.tick_params(axis="y", labelsize=7)
        ax.title.set_fontsize(9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[1, 1].axvline(init_cos.shape[1] - 0.5, color="black", lw=1)
    svals = np.array(results["singular_values"])
    axes[1, 2].bar(range(1, len(svals) + 1), svals**2 / (svals**2).sum(), color="C0")
    axes[1, 2].set(title=f"spectrum of the final displacements (fraction of the variance per direction);\n"
                         f"participation ratio {results['summary']['participation ratio']:.2f} of {n}", xlabel="principal direction")
    axes[1, 2].title.set_fontsize(9)
    fig_title(fig, title)
    fig.savefig(out_dir / "final.png", dpi=150)
    if show:
        plt.show()
    plt.close("all")


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--n-models", type=int, default=8, show_default=True, help="perturbed models (directions 1, ..., n)")
@click.option("--eps", type=float, default=1e-3, show_default=True, help="operator norm of the perturbation of every tensor")
@click.option("--steps", type=int, default=None, help="train this many steps from `start` (default: n_steps - start)")
@click.option("--direction", type=click.Choice(list(DIRECTIONS)), default="randn", show_default=True, help="how the directions are sampled")
@click.option("--traj-every", type=int, default=None, help="keep the displacements for the trajectory plot every this many steps (default: ~50 points)")
@click.option("--n-test", type=int, default=1, show_default=True, help="fixed test examples the losses are compared on")
@click.option("--n-eigs", type=int, default=2, show_default=True, help="top and most negative Hessian eigenvectors the directions are compared to")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--replot", is_flag=True, help="only redo the plots from results.json")
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dir, n_models, eps, steps, direction, traj_every, n_test, n_eigs, device, replot, show):
    exp = Experiment(exp_dir, device)
    if exp.substeps != 1:
        raise click.UsageError(f"{exp.cfg.optimizer} isn't comparable step by step with the other optimizers")
    steps = exp.cfg.n_steps - exp.cfg.start if steps is None else steps
    out_dir = exp_dir / "drift" / f"{direction}_eps{eps:g}_n{n_models}_steps{steps}"
    if replot:
        results = json.loads((out_dir / "results.json").read_text())
    else:
        results = run(exp, out_dir, n_models, eps, steps, direction, traj_every or max(1, steps // 50), n_test, n_eigs)
    plot(out_dir, results, exp.name, show)


if __name__ == "__main__":
    main()
