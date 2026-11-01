"""

Run a whole experiment on DIR (created with `python main.py new DIR ...`) and evaluate it:
  python -m experiments.sweep DIR --kind drop --alphas 0.01,0.02,0.05 --n-train-subsets 10 --n-eval-subsets 5

1. experiments/tune_lr.py if the config has no lr
2. `train`: the full model (if missing), with the default logs
3. `retrain`: n-train-subsets perturbed models per alpha (one vmapped pass)
4. `magic` with n-eval-subsets eval subsets of n-test test examples, once per --variant (repeatable): "hessian" (the plain
   backward, DIR/magic), "gn" (the Gauss-Newton matrix at every step), "gn<T>" (at the steps t < T, the Hessian above),
   "skip<T>" (the steps t < T contribute nothing) or combined, e.g. "gn300_skip100" (into DIR/magic_<variant>)
5. experiments/evaluate.py: the scatter grid per alpha, the pooled ccc / spearman per alpha and their plot vs. alpha
   (DIR/evaluation, and DIR/evaluation_<variant> for the other variants)
Running it again on the same DIR only trains / differentiates what is missing (new alphas, more subsets), and the
evaluation uses everything saved in DIR.
"""

import json
from pathlib import Path

import click

from experiments.evaluate import evaluate
from experiments.tune_lr import tune_lr
from main import HESSIAN, main as cli


@click.command(help=__doc__)
@click.argument("exp_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--kind", type=click.Choice(["drop", "randn"]), default="randn", show_default=True)
@click.option("--alphas", type=str, default="0.001,0.01", show_default=True, help="comma-separated perturbation sizes")
@click.option("--n-train-subsets", type=int, default=20, show_default=True, help="retrained models per alpha")
@click.option("--n-eval-subsets", type=int, default=5, show_default=True, help="eval subsets")
@click.option("--n-test", type=int, default=1, show_default=True, help="test examples in each eval subset")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--compile/--no-compile", default=False, help="torch.compile the plain training passes")
@click.option("--vmap-limit", type=int, default=1, help="retrained models per vmapped pass (default: all of an alpha; 1 is faster for convs, whose vmap is a slow grouped conv)")
@click.option("--variant", "variants", type=str, multiple=True, default=(HESSIAN,), show_default=True,
              help="magic variants, each run and evaluated separately: hessian, gn, gn<T>, skip<T>, gn<T>_skip<T> (see main.py magic --variant)")
@click.option("--show/--no-show", default=True, help="show the final plots (every step's plots are saved in DIR)")
def main(exp_dir, kind, alphas, n_train_subsets, n_eval_subsets, n_test, device, compile, vmap_limit, variants, show):
    exp_dir = str(exp_dir)
    common = ["--device", device, "--compile" if compile else "--no-compile"]
    if json.loads((Path(exp_dir) / "config.json").read_text())["lr"] is None:
        tune_lr(Path(exp_dir), [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2], device=device, compile=compile, show=False)
    alpha_args = [arg for alpha in alphas.split(",") for arg in ("--alpha", alpha)]
    cli(["train", exp_dir, "--no-show", *common], standalone_mode=False)
    cli(["retrain", exp_dir, "--kind", kind, *alpha_args, "--n-subsets", str(n_train_subsets), "--vmap-limit", str(vmap_limit or n_train_subsets), *common], standalone_mode=False)
    for variant in variants:
        cli(["magic", exp_dir, "--n-test", str(n_test), "--n-subsets", str(n_eval_subsets), "--variant", variant, "--no-show", *common], standalone_mode=False)
        evaluate(Path(exp_dir), device, show, variant=variant)


if __name__ == "__main__":
    main()
