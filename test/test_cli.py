import json

from click.testing import CliRunner

from src.cli import cli


def test_cli_end_to_end(tmp_path):
    """train-subsets -> run-magic -> evaluate on a tiny CPU config, with momentum.
    MAGIC's predictions should agree with the retrained models."""
    config = ["--n-steps", "20", "--ckpt-frac", "0.5", "--momentum", "0.9", "--device", "cpu", "--out-dir", str(tmp_path)]
    runner = CliRunner()
    for args in [
        ["train-subsets", "--labels", "0,1", "--n-seeds", "3", "--no-show"],
        ["run-magic", "--eval-labels", "all,0"],
        ["evaluate", "--no-show"],
    ]:
        result = runner.invoke(cli, args + config, catch_exceptions=False)
        assert result.exit_code == 0, result.output

    results = json.loads((tmp_path / "evaluation" / "step_20.json").read_text())
    assert len(results["runs"]) == 6
    for m in results["metrics"]:
        assert m["pearson"] > 0.9
