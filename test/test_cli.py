import json

from click.testing import CliRunner
import torch

from src.cli import main


def test_cli_end_to_end(tmp_path):
    """train-subsets -> run-magic -> evaluate on a tiny CPU config, with momentum.
    MAGIC's predictions should agree with the retrained models."""
    config = ["--n-steps", "20", "--ckpt-frac", "0.5", "--momentum", "0.9", "--device", "cpu", "--out-dir", str(tmp_path)]
    runner = CliRunner()
    for args in [
        ["train-subsets", "--labels", "0,1", "--n-seeds", "3", "--full", "--num-top-eigs", "2", "--no-show"],
        ["run-magic", "--labels", "all,0", "--no-show"],
        ["evaluate", "--no-show"],
    ]:
        result = runner.invoke(main, args + config, catch_exceptions=False)
        assert result.exit_code == 0, result.output

    results = json.loads((tmp_path / "evaluation" / "step_20.json").read_text())
    assert len(results["runs"]) == 6  # the full model isn't one of the retrained runs
    assert (tmp_path / "model_states" / "ckpt_20.pt").exists()
    assert "eig/values" in torch.load(tmp_path / "model_states" / "metrics.pt", weights_only=True)
    assert "delta/buffer" in torch.load(tmp_path / "magic" / "step_20" / "metrics.pt", weights_only=True)
    assert not (tmp_path / "runs" / results["runs"][0] / "metrics.pt").exists()  # retrained runs aren't logged
    for m in results["metrics"]:
        assert m["ccc"] > 0.9
