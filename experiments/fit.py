"""

Fit df/dw by linear regression on the retrained models: a baseline for MAGIC that trains nothing.
  python -m experiments.fit DIR [--n-test 1] [--n-subsets 1] [--perturb-steps k] [--folds 8] [--ridge 0] [--device cpu]

For each eval set (the same `n_test` test examples as `main.py magic`), the real change f(retrained) - f(full model) of its
loss on every retrained model with the given --perturb-steps (DIR/retrained) is regressed on the model's weight perturbation
w - 1, restricted to the examples of the perturbed steps (df/dw = 0 on the others): least squares through the origin, like
MAGIC's prediction betas @ (w - 1) (the minimum-norm solution if there are fewer models than perturbed examples; --ridge: ridge
regression). Saves, in the layout of DIR/magic, DIR/fit[_steps<k>]/<n>_examples_<i>/linearization.pt = {indices, output (f on
the full model), beta [N] (fitted on all the models), cv_preds {retrained model: its predicted change by the fit on the other
--folds - 1 folds}} and examples.png, and prints the in-sample and cross-validated R^2 of each eval set. --folds 0: no
cross-validation, into DIR/fit_insample[_steps<k>] without cv_preds, so evaluate.py --method fit_insample scores the fit on all the
models on those same models.

experiments/evaluate.py --method fit scores the cross-validated predictions, not the in-sample ones: fitting n models on p
perturbed examples explains a fraction p / n of pure noise in-sample (R^2 = 0.5 for n = 2 p of noise), and with n models the
held-out prediction has about (1 + p / (n - p)) times the noise variance of the models themselves, so n >> p for a tight
baseline (n = 2 p is valid but noisy).
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from tqdm import tqdm

from main import Experiment, has_model, magic_dirname, perturb_steps_option, perturbed_steps, plot_examples, show_option


def solve(X: Tensor, Y: Tensor, ridge: float = 0.0) -> Tensor:
    """argmin_B ||X B - Y||^2 + ridge ||B||^2 for X [n, p], Y [n, m] -> [p, m] (the minimum-norm least squares if ridge = 0)"""
    if ridge > 0:
        return torch.linalg.solve(X.T @ X + ridge * torch.eye(X.shape[1], dtype=X.dtype), X.T @ Y)
    return torch.linalg.lstsq(X, Y, driver="gelsd").solution


def r2(pred: Tensor, real: Tensor) -> Tensor:
    """1 - ||pred - real||^2 / ||real||^2 per column of [n, m] (through the origin, like the fit)"""
    return 1 - ((pred - real) ** 2).sum(0) / (real**2).sum(0)


def fit(
    exp_dir: Path, *, n_test: int = 1, n_subsets: int = 1, perturb_steps: int | None = None, folds: int = 8, ridge: float = 0.0,
    device: str = "cpu", show: bool = True,
) -> Path:
    exp = Experiment(exp_dir, device)
    perturbed = perturbed_steps(exp, perturb_steps)
    out_dir = exp_dir / magic_dirname(False, perturb_steps).replace("magic", "fit" if folds else "fit_insample", 1)
    runs = {d: torch.load(d / "perturbation.pt", weights_only=True)
            for d in sorted((exp_dir / "retrained").glob("*")) if has_model(d)}
    runs = {d: p for d, p in runs.items() if p["perturb_steps"] == perturb_steps}
    if not runs:
        raise click.UsageError(f"no retrained models with perturb_steps {perturb_steps} in {exp_dir / 'retrained'}: run `python main.py retrain`")
    idx = torch.cat([exp.loader.batch_indices(t) for t in perturbed]).unique()  # the perturbed examples (the regressors)
    indices = [exp.test_ds.subset(n_test, index=i) for i in range(n_subsets)]  # the test examples of each eval set
    n, p = len(runs), len(idx)
    folds = min(folds, n)
    click.echo(f"fitting {n_subsets} eval sets of {n_test} examples on {n} retrained models x {p} perturbed examples "
               f"({f'{folds}-fold cross-validation' if folds else 'in-sample, no cross-validation'})")

    f_full = exp.eval_losses(exp_dir / "full_model", indices).double()  # [n_fns] (from losses.json where saved, else the state)
    X = torch.stack([pert["weights"] - 1 for pert in runs.values()])[:, idx].double()  # [n, p]
    Y = torch.stack([exp.eval_losses(d, indices).double() - f_full for d in tqdm(runs, desc="retrained models")])  # [n, n_fns]
    beta = solve(X, Y, ridge)  # [p, n_fns]
    cv_preds = torch.zeros_like(Y)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    for k in range(folds):
        held_out = perm[k::folds]
        train = torch.ones(n, dtype=torch.bool)
        train[held_out] = False
        cv_preds[held_out] = X[held_out] @ solve(X[train], Y[train], ridge)
    in_sample, cross_validated = r2(X @ beta, Y), (r2(cv_preds, Y) if folds else None)

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, test_idx in enumerate(indices):
        d = out_dir / f"{n_test}_examples_{i}"
        d.mkdir(exist_ok=True)
        full_beta = torch.zeros(len(exp.train_ds), dtype=torch.get_default_dtype())
        full_beta[idx] = beta[:, i].to(full_beta.dtype)
        lin = {"indices": test_idx, "output": f_full[i].to(full_beta.dtype), "beta": full_beta}
        if folds:
            lin["cv_preds"] = {run.name: cv_preds[r, i].to(full_beta.dtype) for r, run in enumerate(runs)}
        torch.save(lin, d / "linearization.pt")
        click.echo(f"  {n_test} examples #{i}: f = {f_full[i].item():.4f}, R^2 in-sample {in_sample[i].item():.3f}"
                   + ("" if cross_validated is None else f", cross-validated {cross_validated[i].item():.3f}"))
        plot_examples(exp, d, method="linear fit")
    if show:
        plt.show()
    plt.close("all")
    return out_dir


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--n-test", type=int, default=1, show_default=True, help="test examples in each eval set (a fixed random subset): f is their mean loss")
@click.option("--n-subsets", type=int, default=1, show_default=True, help="eval sets (indices 0, ..., n-1)")
@perturb_steps_option
@click.option("--folds", type=int, default=8, show_default=True, help="cross-validation folds of the saved predictions (0: none, in-sample fit into DIR/fit_insample)")
@click.option("--ridge", type=float, default=0.0, show_default=True, help="ridge penalty (0: plain least squares)")
@click.option("--device", type=str, default="cpu", show_default=True)
@show_option
def main(exp_dir, n_test, n_subsets, perturb_steps, folds, ridge, device, show):
    fit(exp_dir, n_test=n_test, n_subsets=n_subsets, perturb_steps=perturb_steps, folds=folds, ridge=ridge, device=device, show=show)


if __name__ == "__main__":
    main()
