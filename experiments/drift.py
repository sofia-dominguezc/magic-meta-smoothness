"""

Is training chaotic or structured at an unstable point? Train copies of the experiment's model (DIR/full_model/start.pt,
or the init if start = 0) that differ by a small perturbation, vmapped together, and track how they drift apart:
  python -m experiments.drift DIR [--n-models 8] [--eps 1e-3] [--steps n] [--direction randn] [--device cpu] [--no-show]

Models (rows of the vmapped training): "center" (unperturbed), "1" ... "n" (center + eps * direction_i, every tensor of
direction_i has operator norm 1, so each tensor moves by eps in the norm of the norm plots; --direction: how the
directions are sampled, `DIRECTIONS`) and "ref" (the center trained plain, unvmapped: its difference to "center" is the
fp32 rounding of the two kernel paths, the noise floor the perturbations are compared to). Only the parameters are
perturbed, not the optimizer buffers. Distances between models are the max over tensors of the operator norm of the
difference (= eps at the start).

Outputs in DIR/drift/<direction>_eps<eps>_n<n>_steps<steps>/ (--replot redoes the analysis and plots from models.pt):
- models.pt: the final states (batched + ref) and the checkpoints every --ckpt-every steps, and the directions
- drift.png: (1) distance of every model to the center over the steps (log scale), with ref; (2) their losses on
  --n-test fixed test examples; (3) trajectories of the displacements from the center, projected on the 2 principal
  directions of the final displacements (a fan = structure along few directions; a cloud = chaos)
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
from dataclasses import asdict
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
from src.data import EvalLoader, Subset
from src.eigvals import top_hessian_eigs
from src.magic import Magic, model_call


def spec_norm(t: Tensor) -> Tensor:
    """Exact operator norm (the norm of `logging.op_norm`): L2 for vectors, spectral of the [out, -1] matrix."""
    return t.norm() if t.ndim <= 1 else torch.linalg.matrix_norm(t.flatten(1), ord=2)


def randn_direction(params: dict[str, Tensor], index: int) -> dict[str, Tensor]:
    """Gaussian, each tensor normalized to operator norm 1 (like `tangent` in `Magic.train_logged`)."""
    g = torch.Generator().manual_seed(index)
    d = {k: torch.randn(p.shape, generator=g, dtype=p.dtype) for k, p in params.items()}
    return {k: t / spec_norm(t) for k, t in d.items()}


DIRECTIONS: dict[str, Callable[[dict[str, Tensor], int], dict[str, Tensor]]] = {"randn": randn_direction}


def dist(a: dict[str, Tensor], b: dict[str, Tensor]) -> float:
    return max(spec_norm(a[k].double() - b[k].double()).item() for k in a)


def flat(params: dict[str, Tensor]) -> Tensor:
    return torch.cat([p.double().flatten() for p in params.values()])


def example_losses(magic: Magic, params: dict[str, Tensor], loader: EvalLoader) -> Tensor:
    """Per-example losses [len(loader)] of the model `params` (on CPU)."""
    params = {k: p.to(magic.device) for k, p in params.items()}
    with torch.no_grad():
        return torch.cat([magic.loss_fn(model_call(magic.model, params, X, bf16=magic.bf16), Y).cpu() for X, Y in loader])


def train(exp: Experiment, out_dir: Path, n_models: int, eps: float, steps: int, direction: str, ckpt_every: int):
    """Train center + n perturbed models (vmapped) and the plain reference; save them to out_dir/models.pt."""
    cfg = exp.cfg
    model, init_state = exp.build()
    magic = exp.magic(model)
    start_state = exp.start_state(magic, init_state).apply(lambda t: t.cpu())
    directions = [DIRECTIONS[direction](start_state.params, i) for i in range(1, n_models + 1)]
    batched = start_state.apply(lambda t: t.expand(n_models + 1, *t.shape).clone())
    for i, d in enumerate(directions, 1):
        for k, t in d.items():
            batched.params[k][i] += eps * t
    click.echo(f"training the reference and {n_models + 1} models (vmapped) for {steps} steps from step {cfg.start}")
    ref_final, ref_ckpts = magic.train(start_state, n_steps=steps, start=cfg.start, ckpt_every=ckpt_every)
    final, ckpts = magic.train(
        batched, n_steps=steps, start=cfg.start, ckpt_every=ckpt_every, weights=torch.ones(n_models + 1, len(exp.train_ds))
    )
    torch.cuda.empty_cache()  # release the trainings' peak (kept reserved by the caching allocator; a no-op on cpu)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "steps": [cfg.start + i * ckpt_every for i in range(len(ckpts))] + [cfg.start + steps],
        "states": [asdict(s) for s in [*ckpts, final]], "ref_states": [asdict(s) for s in [*ref_ckpts, ref_final]],
        "directions": directions, "eps": eps,
    }, out_dir / "models.pt")


def analyze(exp: Experiment, out_dir: Path, n_test: int, n_eigs: int, show: bool) -> dict:
    """Everything in results.json and the plots, from out_dir/models.pt."""
    cfg = exp.cfg
    saved = torch.load(out_dir / "models.pt", weights_only=True)
    model, init_state = exp.build()
    magic = exp.magic(model)
    start_state = exp.start_state(magic, init_state)
    states = [type(init_state)(**s) for s in saved["states"]]
    refs = [type(init_state)(**s) for s in saved["ref_states"]]
    steps, directions, eps = np.array(saved["steps"]), saved["directions"], saved["eps"]
    n = len(directions)
    rows = ["center", *map(str, range(1, n + 1)), "ref"]
    test_loader = exp.eval_loader(Subset(n=n_test))

    def params_of(state, i) -> dict[str, Tensor]:  # row i of a batched state
        return {k: p[i] for k, p in state.params.items()}

    # over the steps: distances to the center and test losses, [n_ckpts, n + 1] (last column: ref)
    dists = np.array([[dist(params_of(s, i), params_of(s, 0)) for i in range(1, n + 1)] + [dist(r.params, params_of(s, 0))]
                      for s, r in zip(states, refs)])
    losses = np.array([[example_losses(magic, params_of(s, i), test_loader).mean().item() for i in range(n + 1)]
                       + [example_losses(magic, r.params, test_loader).mean().item()] for s, r in zip(states, refs)])
    center = flat(params_of(states[-1], 0))
    disp = torch.stack([flat(params_of(states[-1], i)) - center for i in range(1, n + 1)])  # final displacements [n, P]
    disp_ref = flat(refs[-1].params) - center
    # trajectories in the plane of the top 2 principal directions of the final displacements
    _, svals, Vh = torch.linalg.svd(disp, full_matrices=False)
    traj = np.array([[((flat(params_of(s, i)) - flat(params_of(s, 0))) @ Vh[:2].T).numpy() for i in range(1, n + 1)]
                     + [((flat(r.params) - flat(params_of(s, 0))) @ Vh[:2].T).numpy()] for s, r in zip(states, refs)])  # [n_ckpts, n + 1, 2]

    # final models: pairwise distances in parameter and function space, cosines of the displacements
    finals = [params_of(states[-1], i) for i in range(n + 1)] + [refs[-1].params]
    pdist = np.array([[dist(a, b) for b in finals] for a in finals])
    ell = torch.stack([example_losses(magic, p, test_loader) for p in finals]).double().numpy()  # [n + 2, n_test]
    fdist = np.sqrt(((ell[:, None] - ell[None]) ** 2).mean(-1))
    D = torch.cat([disp, disp_ref[None]])
    Dn = D / D.norm(dim=1, keepdim=True)
    cos = (Dn @ Dn.T).numpy()
    clustered = leaves_list(linkage(squareform(pdist[1:-1, 1:-1], checks=False), "average", optimal_ordering=True)) if n > 1 else [0]
    order = [0, *(1 + np.array(clustered)).tolist(), n + 1]  # center, perturbed in clustered order, ref

    # the directions vs. the Hessian at the start state
    X, Y, _ = magic.loader.get(cfg.start)
    params = {k: p.to(magic.device).requires_grad_() for k, p in start_state.params.items()}
    loss = magic.loss_fn(model_call(magic.model, params, X, bf16=magic.bf16), Y).mean()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    eig_names, eigvecs, eigvals = [], [], []
    for negative in [False, True]:
        vals, vecs, _ = top_hessian_eigs(grads, params, n_eigs, precond=start_state.precond(cfg.start), negative=negative)
        eig_names += [f"eig {'-' if negative else ''}{i + 1}" for i in range(n_eigs)]
        eigvecs.append(vecs.detach().cpu().double())
        eigvals += (vals.detach().cpu() * (2 / start_state.stability_threshold)).tolist()
    V = torch.cat(eigvecs, dim=1)  # [P, 2 n_eigs]
    dirs = torch.stack([flat(d) for d in directions])
    eig_cos = {
        "initial direction": ((dirs / dirs.norm(dim=1, keepdim=True)) @ V).numpy(),  # [n, 2 n_eigs]
        "final displacement": (Dn @ V).numpy(),  # [n + 1, 2 n_eigs]
    }

    results = {
        "eps": eps, "steps": steps.tolist(), "rows": rows, "order": order,
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
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    for k, v in results["summary"].items():
        click.echo(f"  {k}: {v:.4g}")
    plot(out_dir, results, exp.name, show)
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
    fig.suptitle(title, fontsize=10)
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
    fig.suptitle(title, fontsize=10)
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
@click.option("--ckpt-every", type=int, default=None, help="checkpoint (and so plot) every this many steps (default: ~50 points)")
@click.option("--n-test", type=int, default=10, show_default=True, help="fixed test examples the losses are compared on")
@click.option("--n-eigs", type=int, default=2, show_default=True, help="top and most negative Hessian eigenvectors the directions are compared to")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--replot", is_flag=True, help="only redo the analysis and plots from models.pt")
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dir, n_models, eps, steps, direction, ckpt_every, n_test, n_eigs, device, replot, show):
    exp = Experiment(exp_dir, device)
    steps = exp.cfg.n_steps - exp.cfg.start if steps is None else steps
    out_dir = exp_dir / "drift" / f"{direction}_eps{eps:g}_n{n_models}_steps{steps}"
    if not replot:
        train(exp, out_dir, n_models, eps, steps, direction, ckpt_every or max(1, steps // 50))
    analyze(exp, out_dir, n_test, n_eigs, show)


if __name__ == "__main__":
    main()
