import json

from click.testing import CliRunner
import torch

from experiments.evaluate import evaluate
from experiments.tune_lr import tune_lr
from main import main
from train_logged import main as train_logged


def test_cli_end_to_end(tmp_path):
    """new -> tune_lr -> train (twice) -> train_logged -> magic -> evaluate on a tiny CPU config, with momentum.
    MAGIC's predictions should agree with the retrained models (in float64: the loss changes are tiny)."""
    d = str(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["new", d, "--dataset", "majority", "--model", "mlp", "--n-steps", "12", "--optimizer", "momentum", "--dtype", "fp64"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert tune_lr(tmp_path, lrs=[0.05, 0.1], steps=6, show=False) in (0.05, 0.1)
    assert json.loads((tmp_path / "config.json").read_text())["lr"] in (0.05, 0.1)
    for args in [
        ["train", d, "--kind", "drop", "--alpha", "0.025", "--alpha", "0.05", "--n-subsets", "2", "--vmap-limit", "3", "--no-show"],  # the full model + 4 models in 2 chunks
        ["train", d, "--kind", "drop", "--alpha", "0.025", "--n-subsets", "4", "--no-show"],  # 2 new models
        ["magic", d, "--n-test", "5", "--n-subsets", "2", "--vmap-limit", "1", "--no-show"],  # 2 eval sets in 2 chunks
        ["magic", d, "--n-test", "5", "--n-subsets", "3", "--no-show"],  # one new eval subset
        ["magic", d, "--n-test", "20", "--no-show"],  # another size (4 eval fns in all)
        ["magic", d, "--n-test", "5", "--no-show"],  # nothing new
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    result = runner.invoke(train_logged, [d, "--num-top-eigs", "2", "--num-neg-eigs", "1", "--eig-every", "2", "--eval-every", "4", "--n-test", "50", "--no-show"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    results = evaluate(tmp_path, show=False)

    assert [(g["kind"], g["alpha"], len(g["runs"])) for g in results["groups"]] == [("drop", 0.025, 4), ("drop", 0.05, 2)]
    assert [(s["n"], s["dir"]) for s in results["eval_sets"]] == [(5, "5_examples_0"), (5, "5_examples_1"), (5, "5_examples_2"), (20, "20_examples_0")]
    lin = torch.load(tmp_path / "magic" / "20_examples_0" / "linearization.pt", weights_only=True)
    assert set(lin) == {"indices", "output", "beta"} and len(lin["indices"]) == 20
    assert (tmp_path / "full_model" / "final.pt").exists() and not (tmp_path / "full_model" / "start.pt").exists()
    full_metrics = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)  # the default logs only
    assert {"loss", "acc", "margin", "params/out.weight", "grads/out.weight"} <= set(full_metrics) and "eig/1" not in full_metrics
    assert len(full_metrics["loss"]["step"]) == 12
    logged = torch.load(tmp_path / "logged" / "metrics.pt", weights_only=True)
    assert {"loss", "eig/1", "eig/2", "eig/-1", "eigvec/-1/out.weight", "ema_eig/1", "loss/ema", "grads/out.weight", "test_loss", "test_loss/ema"} <= set(logged)
    assert "tangent/buffer/out.weight" not in logged  # momentum: no tangent
    assert len(logged["test_loss"]["step"]) == 4 and len(logged["eig/-1"]["step"]) == 6  # steps 0, 4, 8, 12 and 0, 2, ..., 10
    magic_metrics = torch.load(tmp_path / "magic" / "metrics.pt", weights_only=True)  # of the last magic call that computed something
    assert "delta/0/buffer/out.weight" in magic_metrics and len(magic_metrics) == 2 * 6 + 1  # 1 subset: (params, buffer) x 6 tensors + d_weights
    assert not (tmp_path / "retrained" / results["groups"][0]["runs"][0] / "metrics.pt").exists()  # retrained runs aren't logged
    for g in results["groups"]:
        assert g["pooled"]["ccc"] > 0.9 and g["pooled"]["cosine"] > 0.9
        if len(g["runs"]) >= 3:  # the per-set metrics of a 2-run group are degenerate (2 points)
            assert all(m["cosine"] > 0.9 for m in g["metrics"])
        assert (tmp_path / "evaluation" / f"{g['kind']}_{g['alpha']:g}.png").exists()
    assert (tmp_path / "evaluation" / "alpha.png").exists() and (tmp_path / "evaluation" / "results.json").exists()


def test_start_and_randn(tmp_path):
    """Noising the weights only from step 6 on, evaluated at step 12: the state at step 6 is trained once (full_model/start.pt),
    the full model, its logged run and the retrained models start there, and magic reuses it."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--loss", "mse", "--lr", "0.1", "--n-steps", "12", "--start", "6", "--dtype", "fp64"],
        ["train", d, "--kind", "randn", "--alpha", "0.01", "--n-subsets", "3", "--no-show"],
        ["magic", d, "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    (tmp_path / "full_model" / "final.pt").unlink()
    result = runner.invoke(train_logged, [d, "--eval-every", "0", "--no-show"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert set(torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)) == {"params", "lr", "weight_decay"}  # re-saved by train_logged
    logged = torch.load(tmp_path / "logged" / "metrics.pt", weights_only=True)
    assert "tangent/params/out.weight" in logged and logged["loss"]["step"][0] == 6 and "test_loss" not in logged  # SGD: backward AD tangent
    pert = torch.load(tmp_path / "retrained" / "randn_0.01_2" / "perturbation.pt", weights_only=True)
    assert (tmp_path / "retrained" / "randn_0.01_2" / "final.pt").exists() and (pert["kind"], pert["alpha"]) == ("randn", 0.01)
    assert pert["weights"].shape == (512,) and not torch.equal(pert["weights"], torch.ones(512))
    assert set(torch.load(tmp_path / "full_model" / "start.pt", weights_only=True)) == {"params", "lr", "weight_decay"}
    assert torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)["loss"]["step"][0] == 6  # no log before `start`
    results = evaluate(tmp_path, show=False)
    assert [(g["kind"], g["alpha"], len(g["runs"])) for g in results["groups"]] == [("randn", 0.01, 3)]
    assert all(m["ccc"] > 0.9 for m in results["groups"][0]["metrics"])
    assert not (tmp_path / "evaluation" / "alpha.png").exists()  # a single (kind, alpha)


def test_sharpness(tmp_path):
    """The sharpness optimizer (SGD with the step size 2 / top Hessian eigenvalue, a second-order step with aux state) runs
    through train (the full model only: not vmapped), magic and train_logged."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--optimizer", "sharpness", "--lr", "1", "--n-steps", "6", "--dtype", "fp64"],
        ["train", d, "--n-subsets", "0", "--no-show"],
        ["magic", d, "--n-test", "5", "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    result = runner.invoke(train_logged, [d, "--eval-every", "3", "--n-test", "20", "--no-show"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    final = torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)
    assert set(final) == {"params", "lr", "weight_decay", "eigvec", "stats"} and torch.isfinite(final["stats"]["sharpness"])
    lin = torch.load(tmp_path / "magic" / "5_examples_0" / "linearization.pt", weights_only=True)
    assert torch.isfinite(lin["beta"]).all() and (lin["beta"] != 0).any()
    logged = torch.load(tmp_path / "logged" / "metrics.pt", weights_only=True)
    assert "tangent/params/out.weight" not in logged and torch.isfinite(logged["eig/1"]["value"][1:]).all()  # (eig/1 at step 0: P = 0)
    magic_metrics = torch.load(tmp_path / "magic" / "metrics.pt", weights_only=True)
    assert not any(k.startswith("delta/5 examples #0/eigvec") for k in magic_metrics)  # no norms of the aux groups


def test_bf16(tmp_path):
    """fp32 states with the model forward in bfloat16 run through logging (double backward, LOBPCG), MAGIC and the evaluation."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--lr", "0.1", "--n-steps", "4", "--dtype", "bf16", "--lr-schedule", "cosine", "--warmup", "2"],
        ["train", d, "--n-subsets", "2", "--no-show"],
        ["magic", d, "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    result = runner.invoke(train_logged, [d, "--eval-every", "2", "--n-test", "50", "--no-show"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "lr" in torch.load(tmp_path / "logged" / "metrics.pt", weights_only=True)
    final = torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)["params"]
    assert all(p.dtype == torch.float32 for p in final.values())
    lin = torch.load(tmp_path / "magic" / "10_examples_0" / "linearization.pt", weights_only=True)
    assert torch.isfinite(lin["beta"]).all() and (lin["beta"] != 0).any()
    results = evaluate(tmp_path, show=False)
    assert len(results["groups"][0]["runs"]) == 2


def test_newton(tmp_path):
    """The newton optimizer (damped Newton by MINRES: a second-order step with aux state and an implicit backward) runs
    through train (the full model only: not vmapped), magic and train_logged."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--optimizer", "newton", "--lr", "1", "--damping", "2", "--n-steps", "6", "--dtype", "fp64"],
        ["train", d, "--n-subsets", "0", "--no-show"],
        ["magic", d, "--n-test", "5", "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    result = runner.invoke(train_logged, [d, "--eval-every", "3", "--n-test", "20", "--no-show"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    final = torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)
    assert set(final) == {"params", "lr", "weight_decay", "damping", "rtol", "max_iter", "newton", "eigvec", "stats"}
    assert final["damping"] == 2 and 0 < final["stats"]["iterations"] < 100 and torch.isfinite(final["stats"]["residual"])  # converged
    lin = torch.load(tmp_path / "magic" / "5_examples_0" / "linearization.pt", weights_only=True)
    assert torch.isfinite(lin["beta"]).all() and (lin["beta"] != 0).any()
    logged = torch.load(tmp_path / "logged" / "metrics.pt", weights_only=True)
    assert "tangent/params/out.weight" not in logged and torch.isfinite(logged["eig/1"]["value"]).all()
    magic_metrics = torch.load(tmp_path / "magic" / "metrics.pt", weights_only=True)
    assert not any(k.startswith("delta/5 examples #0/newton") for k in magic_metrics)  # no norms of the aux groups
