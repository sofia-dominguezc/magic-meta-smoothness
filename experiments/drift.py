"""

How well do the first- and second-order changes of the training predict the actual change after a small perturbation of the start state?
  python -m experiments.drift DIR [--n-models 4] [--eps 1e-3] [--direction randn] [--device cpu] [--no-show]
  python -m experiments.drift DIR [same settings] --replot [--scale linear]   # only redo the plots from results.json

For each perturbation i = 1 ... n of the parameters at step `start` (eps * direction_i, every tensor of direction_i has
operator norm 1; only the parameters are perturbed, not the optimizer's other tensors), two trainings to `n_steps`, compared
at every step with the central (unperturbed) model:
- "difference": the perturbed model - the central model
- "first order": the tangent ds_t/ds_0 (eps * direction_i) of the central model, by forward-mode AD through its training steps
- "first + second order": plus half the second derivative of s_t along the perturbation (nested forward-mode AD)

Outputs in DIR/drift/<direction>_eps<eps>_n<n>/:
- drift.png: per perturbation, the L2 norms (all the parameters as one vector) of the difference, the two approximations
  and the difference - each approximation (--scale)
- curvature.png: per perturbation, ||H v|| / ||v|| and v^T H v / v^T v of the Hessian H of the batch loss at the central
  model, along v = the difference and v = the tangent, every --curvature-every steps
- losses.png: the loss on --n-test fixed test examples of all the models, and its change from the central model: actual
  (perturbed model), first order and first + second order (--scale)
- results.json: all of the above as numbers
"""

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Callable

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from main import Experiment
from src.magic import Magic
from src.logging import fig_title
from src.optimizers import dict_zip, OptimState
from src.second_order import SecondOrderState


def spec_norms(D: Tensor) -> Tensor:
    """Exact operator norms of a batch [R, *shape] of parameter-shaped tensors -> [R]"""
    return D.norm(dim=-1) if D.ndim <= 2 else torch.linalg.matrix_norm(D.flatten(2), ord=2)


def randn_direction(params: dict[str, Tensor], index: int) -> dict[str, Tensor]:
    """Gaussian, each tensor normalized to operator norm 1"""
    g = torch.Generator().manual_seed(index)
    d = {k: torch.randn(p.shape, generator=g, dtype=p.dtype) for k, p in params.items()}
    return {k: t / spec_norms(t[None])[0] for k, t in d.items()}


DIRECTIONS: dict[str, Callable[[dict[str, Tensor], int], dict[str, Tensor]]] = {"randn": randn_direction}


def dot(a: dict[str, Tensor], b: dict[str, Tensor]) -> Tensor:
    """Inner product of two dicts of tensors as vectors"""
    return torch.stack([(x.double() * y.double()).sum() for x, y in dict_zip(a, b).values()]).sum()


def taylor(f: Callable, x: list[Tensor], a: list[Tensor], b: list[Tensor]):
    """f(x) and the 1st and 2nd derivatives at eps = 0 of f(x + eps a + eps^2/2 b), by nested forward-mode AD"""
    g = lambda eps: f([x_ + eps * a_ + eps**2 / 2 * b_ for x_, a_, b_ in zip(x, a, b)])

    def dg(eps: Tensor):
        out, d = torch.func.jvp(g, (eps,), (torch.ones_like(eps),))
        return d, (out, d)

    eps = torch.zeros((), dtype=x[0].dtype, device=x[0].device)
    _, d2, (out, d) = torch.func.jvp(dg, (eps,), (torch.ones_like(eps),), has_aux=True)
    return out, d, d2


def taylor_step(magic: Magic, t: int, state: OptimState, a: OptimState, b: OptimState) -> tuple[OptimState, OptimState, OptimState]:
    """Step t of the training on the 2nd-order expansion s_t + eps a_t + eps^2/2 b_t of the state: (s_{t+1}, a_{t+1}, b_{t+1})"""
    X, Y, _ = magic.loader.get_batch(t)

    def step(tensors: list[Tensor]) -> tuple[Tensor, ...]:
        s = state.unflatten(tensors)
        grads = torch.func.grad(lambda params: magic.objective.forward(params, X, Y)[0])(s.params)
        return tuple(s._update(t, grads))

    new, da, db = taylor(step, [x.detach() for x in state], list(a), list(b))
    return state.unflatten(new), state.unflatten(da), state.unflatten(db)


def run(exp: Experiment, out_dir: Path, n_models: int, eps: float, direction: str, n_test: int, every: int) -> dict:
    """Train each perturbed model next to the central model with its tangent, compare them at every step"""
    magic = exp.magic
    start, n_steps = exp.steps
    start_state = exp.start_state.apply(lambda t: t.to(exp.device))
    cpu_params = {k: p.cpu() for k, p in start_state.params.items()}
    X, Y = (x.to(exp.device) for x in exp.test_ds.examples(exp.test_ds.subset(n_test)))
    example_loss = lambda p: magic.objective.forward(p, X, Y)[0]

    runs: dict[str, list[Tensor]] = defaultdict(list)  # name -> per perturbation [steps]
    curv_steps, thresholds = [], []
    for i in range(1, n_models + 1):
        click.echo(f"perturbation {i} of {n_models}")
        d = {k: eps * t.to(exp.device) for k, t in DIRECTIONS[direction](cpu_params, i).items()}
        init = replace(start_state, params={k: p + d[k] for k, p in start_state.params.items()})
        zeros = start_state.apply(torch.zeros_like)
        central, tangent, second = start_state, replace(zeros, params=d), zeros  # s_t + a_t + b_t / 2: 2nd-order expansion in the perturbation
        out: dict[str, list[Tensor]] = defaultdict(list)  # name -> per step
        for t, perturbed, grads in magic.train.trace(init, n_steps=n_steps, start=start):
            center, T = central.params, tangent.params
            D = {k: p.detach() - center[k] for k, p in perturbed.params.items()}
            S = {k: T[k] + second.params[k] / 2 for k in T}  # first + second order
            vectors = {"diff": D, "tangent": T, "second": S, "error": {k: D[k] - T[k] for k in D}, "error2": {k: D[k] - S[k] for k in D}}
            for name, v in vectors.items():
                out[name].append(dot(v, v).sqrt())
            with torch.no_grad():
                loss, d_loss, d2_loss = taylor(lambda p: example_loss(dict(zip(center, p))), list(center.values()), list(T.values()), list(second.params.values()))
                out["center_loss"].append(loss)
                out["loss_tangent"].append(d_loss)
                out["loss_second"].append(d_loss + d2_loss / 2)
                out["loss"].append(example_loss(perturbed.params))
            if grads is None:
                break
            if (t - start) % every == 0:
                hvp_fn = magic.objective.hvp_fn(center, *magic.loader.get_batch(t)[:2])
                for name, v in (("diff", D), ("tangent", T)):
                    Hv = hvp_fn(v)
                    out[f"{name}_Hv"].append((dot(Hv, Hv) / dot(v, v)).sqrt())
                    out[f"{name}_vHv"].append(dot(v, Hv) / dot(v, v))
                if i == 1:
                    curv_steps.append(t)
                    P = next(iter(central.precond(t).values()))
                    thresholds.append(None if isinstance(P, Tensor) else start_state.stability_threshold / P)
            central, tangent, second = taylor_step(magic, t, central, tangent, second)
        for name, values in out.items():
            runs[name].append(torch.stack(values).cpu())

    center_loss = runs.pop("center_loss")[0]
    results = {
        "eps": eps, "direction": direction, "n_test": n_test,
        "steps": list(range(start, start + n_steps + 1)), "curvature_steps": curv_steps,
        "threshold": thresholds, "center_loss": center_loss.tolist(),
        **{name: torch.stack(values, 1).tolist() for name, values in runs.items()},  # [steps, n_models]
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results))
    return results


def plot(out_dir: Path, results: dict, title: str, scale: str, show: bool):
    """drift.png, curvature.png and losses.png from the results of `run`"""
    steps, curv_steps = np.array(results["steps"]), np.array(results["curvature_steps"])
    arrays = {k: np.array(v, dtype=float) for k, v in results.items() if isinstance(v, list) and k not in ("steps", "curvature_steps")}
    arrays = {k: np.where(np.abs(v) < 1e30, v, np.nan) for k, v in arrays.items()}  # (a diverged training)
    n = arrays["diff"].shape[1]
    title = f"{n} models perturbed by eps = {results['eps']:g} ({results['direction']}) at step {steps[0]:g}; {title}"
    positive = lambda a: np.where(a > 0, a, np.nan) if scale == "log" else a
    ncols = min(n, 4)
    nrows = -(-n // ncols)

    def grid():
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows), squeeze=False, layout="constrained")
        for ax in axes.flat[n:]:
            fig.delaxes(ax)
        fig_title(fig, title)
        return fig, axes.flat[:n]

    fig, axes = grid()
    lines = [("diff", "difference: perturbed - central", "C0", "-"), ("tangent", "first order (tangent)", "C1", "--"),
             ("second", "first + second order", "C2", ":"), ("error", "difference - first order", "C1", "-"),
             ("error2", "difference - (first + second order)", "C2", "-")]
    for i, ax in enumerate(axes):
        for key, label, color, ls in lines:
            ax.plot(steps, positive(arrays[key][:, i]), ls, lw=1.2, color=color, label=label)
        ax.set(title=f"perturbation {i + 1}: L2 norm", xlabel="step", yscale=scale)
        ax.title.set_fontsize(9)
        ax.legend(fontsize=7)
    fig.savefig(out_dir / "drift.png", dpi=150)

    fig, axes = grid()
    lines = [("diff_Hv", "difference: ||Hv|| / ||v||", "C0", "-"), ("diff_vHv", "difference: v^T H v / v^T v", "C0", "--"),
             ("tangent_Hv", "tangent: ||Hv|| / ||v||", "C1", "-"), ("tangent_vHv", "tangent: v^T H v / v^T v", "C1", "--")]
    style = "o" if len(curv_steps) < 2 or curv_steps[1] - curv_steps[0] > 1 else ""
    for i, ax in enumerate(axes):
        for key, label, color, ls in lines:
            ax.plot(curv_steps, arrays[key][:, i], style + ls, ms=2.5, lw=1, color=color, label=label)
        if not np.isnan(arrays["threshold"]).all():
            ax.plot(curv_steps, arrays["threshold"], ":", lw=1, color="gray", label="stability threshold")
        ax.set(title=f"perturbation {i + 1}: curvature of the batch loss along v", xlabel="step")
        ax.title.set_fontsize(9)
        ax.legend(fontsize=7)
    fig.savefig(out_dir / "curvature.png", dpi=150)

    cols = min(n + 1, 4)
    fig, axes = plt.subplots(-(-(n + 1) // cols), cols, figsize=(4.8 * cols, 3.6 * -(-(n + 1) // cols)), squeeze=False, layout="constrained")
    axes = axes.flatten()
    for ax in axes[n + 1:]:
        fig.delaxes(ax)
    colors = [plt.cm.viridis(i / max(1, n - 1)) for i in range(n)]
    actual, first_order, second_order = arrays["loss"] - arrays["center_loss"][:, None], arrays["loss_tangent"], arrays["loss_second"]
    if scale == "log":
        actual, first_order, second_order = (positive(np.abs(x)) for x in (actual, first_order, second_order))
    for i in range(n):
        axes[0].plot(steps, arrays["loss"][:, i], lw=1, color=colors[i], label=f"perturbed {i + 1}")
        axes[i + 1].plot(steps, actual[:, i], "-", lw=1.2, color="C0", label="actual: perturbed - central")
        axes[i + 1].plot(steps, first_order[:, i], "--", lw=1.2, color="C1", label="first order (tangent)")
        axes[i + 1].plot(steps, second_order[:, i], ":", lw=1.2, color="C2", label="first + second order")
        axes[i + 1].set(title=f"perturbation {i + 1}: " + ("|change|" if scale == "log" else "change") + " of the test loss", xlabel="step", yscale=scale)
    axes[0].plot(steps, arrays["center_loss"], lw=1.5, color="red", label="central")
    axes[0].set(title=f"test loss ({results['n_test']} fixed examples)", xlabel="step")
    for ax in axes[:n + 1]:
        ax.title.set_fontsize(9)
        ax.legend(fontsize=7)
    fig_title(fig, title)
    fig.savefig(out_dir / "losses.png", dpi=150)
    if show:
        plt.show()
    plt.close("all")


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--n-models", type=int, default=4, show_default=True, help="perturbations (directions 1, ..., n)")
@click.option("--eps", type=float, default=1e-3, show_default=True, help="operator norm of the perturbation of every tensor")
@click.option("--direction", type=click.Choice(list(DIRECTIONS)), default="randn", show_default=True, help="how the directions are sampled")
@click.option("--n-test", type=int, default=1, show_default=True, help="fixed test examples the losses are compared on")
@click.option("--curvature-every", type=int, default=10, show_default=True, help="curvature along the difference and the tangent every this many steps")
@click.option("--scale", type=click.Choice(["log", "linear"]), default="log", show_default=True, help="y scale of the norms and loss changes")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--replot", is_flag=True, help="only redo the plots from results.json")
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dir, n_models, eps, direction, n_test, curvature_every, scale, device, replot, show):
    exp = Experiment(exp_dir, device)
    out_dir = exp_dir / "drift" / f"{direction}_eps{eps:g}_n{n_models}"
    if replot:
        results = json.loads((out_dir / "results.json").read_text())
    else:
        if isinstance(exp.init_state, SecondOrderState):
            raise click.UsageError(f"{exp.cfg.optimizer} isn't supported (first-order optimizers only)")
        results = run(exp, out_dir, n_models, eps, direction, n_test, curvature_every)
    plot(out_dir, results, exp.name, scale, show)
    click.echo(f"saved {out_dir}")


if __name__ == "__main__":
    main()
