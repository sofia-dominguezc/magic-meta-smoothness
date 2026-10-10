import json

from click.testing import CliRunner
import torch

from main import main


def test_cli_end_to_end(tmp_path):
    """new -> tune-lr -> retrain -> run-magic -> evaluate on a tiny CPU config, with momentum.
    MAGIC's predictions should agree with the retrained models."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--size", "small", "--n-steps", "20", "--ckpt-frac", "0.5", "--momentum", "0.9"],
        ["tune-lr", d, "--lrs", "0.05,0.1", "--steps", "10", "--no-show"],
        ["retrain", d, "--alpha", "0.025", "--labels", "0,1", "--n-seeds", "3", "--full", "--num-top-eigs", "2", "--no-show"],  # lr from tune-lr
        ["run-magic", d, "--labels", "all,0", "--no-show"],
        ["evaluate", d, "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output

    assert json.loads((tmp_path / "config.json").read_text())["lr"] in (0.05, 0.1)
    results = json.loads((tmp_path / "evaluation" / "step_20.json").read_text())
    assert len(results["runs"]) == 6  # the full model isn't one of the retrained runs
    assert (tmp_path / "model_states" / "ckpt_20.pt").exists() and (tmp_path / "model_states" / "eos.json").exists()
    full_metrics = torch.load(tmp_path / "model_states" / "metrics.pt", weights_only=True)
    assert {"eig/1", "eig/2", "tangent/buffer/out.weight", "grads/out.weight", "test loss"} <= set(full_metrics)
    assert "delta/test set (seed 0)/buffer/out.weight" in torch.load(tmp_path / "magic" / "step_20" / "metrics.pt", weights_only=True)
    assert not (tmp_path / "runs" / results["runs"][0] / "metrics.pt").exists()  # retrained runs aren't logged
    for m in results["metrics"]:
        assert m["ccc"] > 0.9
    assert results["pooled"]["ccc"] > 0.9

    # noising the weights only from step 10 on, evaluated at step 20 (the state at step 10 is retrained, not a checkpoint)
    for args in [
        ["retrain", d, "--kind", "randn", "--alpha", "0.1", "--start", "10", "--n-seeds", "3", "--no-full"],
        ["run-magic", d, "--start", "10", "--no-show"],
        ["evaluate", d, "--start", "10", "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    results = json.loads((tmp_path / "evaluation" / "step_20_from_10.json").read_text())
    assert len(results["runs"]) == 3 and all(r.startswith("randn_0.1_seed_") and r.endswith("_from_10") for r in results["runs"])
    for m in results["metrics"]:
        assert m["ccc"] > 0.9


def test_mse_experiment(tmp_path):
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--size", "small", "--loss", "mse", "--lr", "0.1", "--n-steps", "4", "--ckpt-frac", "0.5"],
        ["retrain", d, "--n-seeds", "1", "--eval-every", "2", "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    assert (tmp_path / "runs" / "drop_5.0pct_all_labels_seed_0" / "ckpt_4.pt").exists()
