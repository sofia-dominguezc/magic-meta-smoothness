"""
MAGIC forwards: a tangent v_t attached to the training state s_t, propagated by forward-mode AD through the training (the
loss gradient and the step), and compared at every step with what MAGIC ending at t would predict: df(s_t)/ds_t . v_t.
  python -m experiments.magic_eigs DIR [--n-test 10] [--n-subsets 1] [--init zero|eig] [--perturb-steps k] [--kind randn] [--alpha 0.05]
                                       [--index 0] [--hvp-every 1] [--n-eigs 0] [--n-neg-eigs 0] [--eig-every 10] [--device cpu]

The tangent is v_t = ds_t/dw . dw, the first-order effect of the training weight perturbation dw = w - 1 (`Dataset.weights`:
--kind, --alpha, --index, as in `main.py retrain`) on the state, accumulated over the perturbed steps (--perturb-steps k: the
first k from `start`; 0: no perturbation; default: all), plus an initial tangent at `start` (--init: zero, or eig: the top
Hessian eigenvector, of params L2 norm --init-norm). So v_T . df/ds_T = df/dw . dw, the MAGIC prediction of the change in f,
and v_t . df(s_t)/ds_t is the prediction of MAGIC run to step t: one training replaces a MAGIC backward per step.
Saves DIR/magic_eigs/<init>_<perturbation>/metrics.pt (and plots) and tangent.pt = v_T (+ the perturbation):
- "magic/i": df(s_t)/ds_t . v_t per eval set (and "magic/batch": the batch gradient g_t . v_t), "magic_examples/i": the RMS over
  the eval examples of their own df_j/ds_t . v_t, "magic_cos/i": cos(df/ds_t, v_t), "grad_f/i": |df/ds_t|, "f/i": f(s_t)
- "tangent/<group>": |v_t| per state group and its relative change per step ("displacement/tangent/<group>"); "growth/tangent":
  |v_{t+1}| / |v_t| over the params; "tangent_cos/step": cos(v_{t+1}, v_t), "tangent_cos/start": cos(v_t, v_start) (--init eig)
- every --hvp-every steps (one HVP): "eig/tangent_curvature" = v^T H v / v^T v and "eig/tangent_Hv" = |H v| / |v| of the batch
  Hessian H at s_t, and "eig/threshold" = stability threshold / step size (scalar step sizes); plain SGD: "growth/sgd_linear"
  = |(I - lr (H + wd)) v| / |v|, the growth without the step's perturbation
- every --eig-every steps, if --n-eigs / --n-neg-eigs (LOBPCG, warm-started): "eig/i" the top (i > 0) / most negative (i < 0)
  eigenvalues, "tangent_eig/i": |cos| of v_t with the eigenvectors
"""

from dataclasses import replace
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor

from main import Experiment, eig_cos, perturbed_steps, retrained_dirname
from src.lobpcg import hessian_eigs
from src.logging import Logger, total_l2
from src.optimizers import OptimState, SGDState
from src.utils import dict_zip

INITS = ["zero", "eig"]


def dot(u: dict[str, Tensor], v: dict[str, Tensor]) -> Tensor:
    """The dot product of two dicts of tensors as vectors"""
    return torch.stack([(a * b).sum() for a, b in dict_zip(u, v).values()]).sum()


def cos(u: dict[str, Tensor], v: dict[str, Tensor]) -> Tensor:
    return dot(u, v) / (total_l2(u) * total_l2(v))


def magic_eigs(
    exp_dir: Path, *, n_test: int = 1, n_subsets: int = 1, init: str = "zero", init_norm: float = 1.0,
    perturb_steps: int | None = None, kind: str = "randn", alpha: float = 0.05, index: int = 0,
    hvp_every: int = 1, n_eigs: int = 0, n_neg_eigs: int = 0, eig_every: int = 10, eval_every: int = 1,
    device: str = "cpu", compile: bool = False, show: bool = True,
) -> Path:
    exp = Experiment(exp_dir, device, compile)
    magic, loader = exp.magic, exp.loader
    start, n_steps = exp.steps
    perturbed = range(start, start) if perturb_steps == 0 else perturbed_steps(exp, perturb_steps)
    perturbation = "none" if perturb_steps == 0 else retrained_dirname(kind, alpha, index, perturb_steps)
    out_dir = exp_dir / "magic_eigs" / f"{init}_{perturbation}"
    eval_sets = [exp.test_ds.examples(exp.test_ds.subset(n_test, index=i)) for i in range(n_subsets)]
    dw = (exp.train_ds.weights(kind, alpha, index) - 1).to(magic.device)  # [N] the perturbation of the training weights
    ks = {sign: k for sign, k in ((1, n_eigs), (-1, n_neg_eigs)) if k}  # sign -> eigenpairs by LOBPCG
    warm: dict[int, dict[str, Tensor]] = {}  # sign -> eigenvectors [k, *shape] per param, LOBPCG's warm start
    logger = Logger(f"MAGIC forwards: tangent {init} + perturbation {perturbation}; {exp.name}", smooth=False, total_norms=True,
                    step_scale=exp.substeps)
    logger.log_panels.update({"growth", "grad_f", "magic_examples"})
    eos = exp.start_state.stability_threshold  # logged normalized by the step size as "eig/threshold"

    state = exp.start_state.apply(lambda s: s.to(magic.device, copy=True))
    forward, _, _ = magic._train_forward(state, None)
    v = state.apply(torch.zeros_like)  # the tangent v_t, plain tensors
    if init == "eig":
        X, Y, _ = loader.get_batch(start)
        g = torch.Generator().manual_seed(start)
        u = {k: torch.randn(p.shape, generator=g, dtype=p.dtype).to(p.device) for k, p in state.params.items()}
        _, u, _ = hessian_eigs(magic.hvp_fn(state.params, X, Y), u, batched=False)
        v = replace(v, params={k: init_norm * e / total_l2(u) for k, e in u.items()})
    else:
        assert init == "zero", init
    v_start = None if init == "zero" else v.params

    def log_tangent(t: int, state: OptimState, v: OptimState):
        """The metrics of (s_t, v_t) that need no tangents: norms, cos with v_start, the HVP metrics and the eigenvalues"""
        logger.log_norms(t, "tangent", v)
        if v_start is not None:
            logger.log(t, **{"tangent_cos/start": cos(v.params, v_start)})
        if t == start + n_steps:
            return
        hvp_step, eig_step = (t - start) % hvp_every == 0, ks and (t - start) % eig_every == 0
        if not (hvp_step or eig_step):
            return
        X, Y, _ = loader.get_batch(t)
        hvp_fn = magic.hvp_fn(state.params, X, Y)
        if hvp_step:
            Hv = hvp_fn(v.params)
            norm = total_l2(v.params)
            logger.log(t, **{"eig/tangent_curvature": dot(v.params, Hv) / norm**2, "eig/tangent_Hv": total_l2(Hv) / norm})
            if type(state) is SGDState:
                lr = state.lr(t)
                linear = {k: u - lr * (hu + state.wd * u) for k, (u, hu) in dict_zip(v.params, Hv).items()}
                logger.log(t, **{"growth/sgd_linear": total_l2(linear) / norm})
            P = next(iter(state.precond(t).values()))
            if not isinstance(P, Tensor) or P.ndim == 0:
                logger.log(t, **{"eig/threshold": torch.as_tensor(eos / P, dtype=torch.get_default_dtype())})
        if eig_step:
            for sign, k in ks.items():
                if sign not in warm:
                    g = torch.Generator().manual_seed(t)
                    warm[sign] = {name: torch.randn(k, *p.shape, generator=g, dtype=p.dtype).to(p.device) for name, p in state.params.items()}
                eigvals, warm[sign], n_iter = hessian_eigs(hvp_fn, warm[sign], None, sign)
                ids = [sign * (i + 1) for i in range(k)]
                logger.log(t, **{f"eig/{i}": lam for i, lam in zip(ids, eigvals.unbind())}, **{f"lobpcg_iters/{sign}": torch.tensor(n_iter)})
                logger.log(t, **{f"tangent_eig/{i}": c for i, c in zip(ids, eig_cos(v.params, warm[sign]).unbind())})

    def log_eval(t: int, dual: OptimState, params: dict[str, Tensor]):
        """In a dual level: f_i(s_t), its tangent df_i/ds_t . v_t (per example, by the dual forward) and its gradient"""
        for i, (X_i, Y_i) in enumerate(eval_sets):
            loss, out = forward(params, X_i, Y_i)
            f, df = fwAD.unpack_dual(loss)
            grad_f = torch.autograd.grad(f, list(params.values()))
            grad_f = {k: g.detach() for k, g in zip(params, grad_f)}
            df_examples = fwAD.unpack_dual(magic.loss_fn(out, Y_i)).tangent  # [n]
            logger.log(t, **{f"f/{i}": f.detach(), f"magic/{i}": df.detach(), f"grad_f/{i}": total_l2(grad_f),
                             f"magic_examples/{i}": df_examples.detach().pow(2).mean().sqrt(),
                             f"magic_cos/{i}": df.detach() / (total_l2(grad_f) * total_l2(dual.params))})

    for t in range(start, start + n_steps):
        log_tangent(t, state, v)
        X, Y, idx = loader.get_batch(t)
        with fwAD.dual_level():
            dual = state.unflatten([fwAD.make_dual(s, u) for s, u in zip(state, v)])
            params = {k: p.requires_grad_() for k, p in dual.params.items()}
            if eval_every and (t - start) % eval_every == 0:
                log_eval(t, dual, params)
            w = torch.ones_like(idx, dtype=X.dtype)
            if t in perturbed:
                w = fwAD.make_dual(w, dw[idx])
            next_state, grads, loss, _ = dual.step(t, lambda params: forward(params, X, Y, w))
            logger.log(t, loss=fwAD.unpack_dual(loss).primal.detach(),
                       **{"magic/batch": dot({k: fwAD.unpack_dual(g).primal for k, g in grads.items()}, v.params)})
            tangents = [fwAD.unpack_dual(s).tangent for s in next_state]
            tangents = [torch.zeros_like(s) if tan is None else tan.detach() for s, tan in zip(next_state, tangents)]
        state = next_state.apply(lambda s: s.detach())
        v_next = state.unflatten(tangents)
        logger.log(t, **{"growth/tangent": total_l2(v_next.params) / total_l2(v.params), "tangent_cos/step": cos(v_next.params, v.params)})
        v = v_next
    T = start + n_steps
    log_tangent(T, state, v)
    with fwAD.dual_level():
        dual = state.unflatten([fwAD.make_dual(s, u) for s, u in zip(state, v)])
        log_eval(T, dual, {k: p.requires_grad_() for k, p in dual.params.items()})

    out_dir.mkdir(parents=True, exist_ok=True)
    tangent = {group: {k: u.cpu() for k, u in getattr(v, group).items()} for group in v.groups}
    torch.save({"tangent": tangent, "init": init, "init_norm": init_norm, "kind": kind, "alpha": alpha, "index": index,
                "perturb_steps": perturb_steps}, out_dir / "tangent.pt")
    logger.finish(out_dir / "metrics.pt")
    click.echo(f"saved {out_dir / 'metrics.pt'}, its plots and tangent.pt")
    if show:
        plt.show()
    plt.close("all")
    return out_dir


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--n-test", type=int, default=1, show_default=True, help="test examples in each eval set")
@click.option("--n-subsets", type=int, default=1, show_default=True, help="eval sets (a line each)")
@click.option("--init", type=click.Choice(INITS), default="zero", show_default=True, help="the tangent at `start`: zero, or the top Hessian eigenvector")
@click.option("--init-norm", type=float, default=1.0, show_default=True, help="L2 norm of the initial tangent (--init eig)")
@click.option("--perturb-steps", type=int, default=None, help="add the perturbation of the training weights of the first this many steps from `start` only (0: none; default: all the steps)")
@click.option("--kind", type=click.Choice(["drop", "randn"]), default="randn", show_default=True, help="of the perturbation (`main.py retrain`)")
@click.option("--alpha", type=float, default=0.05, show_default=True, help="size of the perturbation")
@click.option("--index", type=int, default=0, show_default=True, help="index of the perturbation")
@click.option("--hvp-every", type=int, default=1, show_default=True, help="steps between the HVP metrics (curvature along the tangent)")
@click.option("--n-eigs", type=int, default=0, show_default=True, help="top eigenvalues (LOBPCG)")
@click.option("--n-neg-eigs", type=int, default=0, show_default=True, help="most negative eigenvalues (LOBPCG)")
@click.option("--eig-every", type=int, default=10, show_default=True, help="steps between eigenvalue computations")
@click.option("--eval-every", type=int, default=1, show_default=True, help="steps between the eval set metrics")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--compile/--no-compile", default=False, help="torch.compile the model forward")
@click.option("--show/--no-show", default=True, help="show the plots")
def main(exp_dir, **kwargs):
    magic_eigs(exp_dir, **kwargs)


if __name__ == "__main__":
    main()
