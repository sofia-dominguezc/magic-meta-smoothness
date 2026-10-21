import json

from click.testing import CliRunner
import torch

from experiments.evaluate import evaluate
from experiments.tune_lr import tune_lr
from main import main


def test_cli_end_to_end(tmp_path):
    """new -> tune_lr -> train -> retrain (twice: two alphas) -> run-magic -> evaluate on a tiny CPU config, with momentum.
    MAGIC's predictions should agree with the retrained models (in float64: the loss changes are tiny)."""
    d = str(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["new", d, "--dataset", "majority", "--model", "mlp", "--n-steps", "12", "--optimizer", "momentum", "--dtype", "fp64"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert tune_lr(tmp_path, lrs=[0.05, 0.1], steps=6, show=False) in (0.05, 0.1)
    assert json.loads((tmp_path / "config.json").read_text())["lr"] in (0.05, 0.1)
    for args in [
        ["train", d, "--num-top-eigs", "2", "--num-neg-eigs", "1", "--eig-every", "2", "--eval-every", "4", "--n-test", "50", "--no-show"],
        ["retrain", d, "--alpha", "0.025", "--n-subsets", "4"],  # + the unperturbed model, full_model/final.pt
        ["retrain", d, "--alpha", "0.05", "--n-subsets", "2"],
        ["run-magic", d, "--n-test", "5", "--n-subsets", "2", "--no-show"],
        ["run-magic", d, "--n-test", "5", "--n-subsets", "3", "--no-show"],  # one new eval subset
        ["run-magic", d, "--n-test", "20", "--no-show"],  # another size (4 eval fns in all)
        ["run-magic", d, "--n-test", "5", "--no-show"],  # nothing new
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    results = evaluate(tmp_path, show=False)

    assert [(g["kind"], g["alpha"], len(g["runs"])) for g in results["groups"]] == [("drop", 0.025, 4), ("drop", 0.05, 2)]
    assert [(s["n"], s["index"]) for s in results["eval_subsets"]] == [(5, 0), (5, 1), (5, 2), (20, 0)]
    assert (tmp_path / "full_model" / "final.pt").exists() and not (tmp_path / "full_model" / "start.pt").exists()
    full_metrics = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)
    assert {"eig/1", "eig/2", "eig/-1", "tangent/buffer/out.weight", "grads/out.weight", "test_loss"} <= set(full_metrics)
    assert len(full_metrics["test_loss"]["step"]) == 3
    magic_metrics = torch.load(tmp_path / "magic" / "metrics.pt", weights_only=True)  # of the last run-magic call that computed something
    assert "delta/20 examples #0/buffer/out.weight" in magic_metrics and len(magic_metrics) == 2 * 6  # 1 subset: (params, buffer) x 6 tensors
    assert not (tmp_path / "retrained" / results["groups"][0]["runs"][0] / "metrics.pt").exists()  # retrained runs aren't logged
    for g in results["groups"]:
        assert all(m["cosine"] > 0.9 for m in g["metrics"]) and g["pooled"]["ccc"] > 0.9  # (the ccc of a 2-run group is degenerate)
        assert (tmp_path / "evaluation" / f"{g['kind']}_{g['alpha']:g}.png").exists()
    assert (tmp_path / "evaluation" / "alpha.png").exists() and (tmp_path / "evaluation" / "results.json").exists()


def test_start_and_randn(tmp_path):
    """Noising the weights only from step 6 on, evaluated at step 12: the state at step 6 is trained once (full_model/start.pt),
    the full model's log and the retrained models start there, and run-magic reuses it."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--loss", "mse", "--lr", "0.1", "--n-steps", "12", "--start", "6", "--dtype", "fp64"],
        ["train", d, "--eval-every", "0", "--no-show"],
        ["retrain", d, "--kind", "randn", "--alpha", "0.01", "--n-subsets", "3"],
        ["run-magic", d, "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    assert (tmp_path / "retrained" / "randn_0.01_2" / "final.pt").exists()
    assert set(torch.load(tmp_path / "full_model" / "start.pt", weights_only=True)) == {"params", "lr", "weight_decay"}
    assert torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)["loss"]["step"][0] == 6  # no log before `start`
    results = evaluate(tmp_path, show=False)
    assert [(g["kind"], g["alpha"], len(g["runs"])) for g in results["groups"]] == [("randn", 0.01, 3)]
    assert all(m["ccc"] > 0.9 for m in results["groups"][0]["metrics"])
    assert not (tmp_path / "evaluation" / "alpha.png").exists()  # a single (kind, alpha)


def test_bf16(tmp_path):
    """fp32 states with the model forward in bfloat16 run through logging (forward-mode AD, LOBPCG), MAGIC and the evaluation."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--lr", "0.1", "--n-steps", "4", "--dtype", "bf16", "--lr-schedule", "cosine", "--warmup", "2"],
        ["train", d, "--eval-every", "2", "--n-test", "50", "--no-show"],
        ["retrain", d, "--n-subsets", "2"],
        ["run-magic", d, "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    final = torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)["params"]
    assert all(p.dtype == torch.float32 for p in final.values())
    lin = torch.load(tmp_path / "magic" / "10_examples_0" / "linearization.pt", weights_only=True)
    assert torch.isfinite(lin["beta"]).all() and (lin["beta"] != 0).any()
    results = evaluate(tmp_path, show=False)
    assert len(results["groups"][0]["runs"]) == 2
