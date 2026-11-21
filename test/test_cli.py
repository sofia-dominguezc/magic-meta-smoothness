import json
from pathlib import Path

import click
import pytest
from matplotlib.figure import Figure

from click.testing import CliRunner
import torch

from experiments.evaluate import evaluate
from experiments.tune_lr import tune_lr
from main import Experiment, main
from src.second_order import CentralFlowState

@pytest.fixture(autouse=True)
def no_rendering(monkeypatch):
    """The plots are built but not drawn (most of a CLI test's time): `savefig` only creates the file"""
    monkeypatch.setattr(Figure, "savefig", lambda self, path, **kwargs: Path(path).touch())


def test_cli_end_to_end(tmp_path):
    """new -> tune_lr -> retrain (twice) -> train -> magic -> evaluate on a tiny CPU config, with momentum.
    MAGIC's predictions should agree with the retrained models (in float64: the loss changes are tiny)."""
    d = str(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["new", d, "--dataset", "majority", "--model", "mlp", "--width", "16", "--n-steps", "12", "--optimizer", "momentum", "--dtype", "fp64"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert tune_lr(tmp_path, lrs=[0.05, 0.1], steps=6, show=False, magic=True, n_test=5) in (0.05, 0.1)
    assert (tmp_path / "lr_tuning_curves.png").exists() and (tmp_path / "lr_tuning_magic.png").exists()
    assert len(json.loads((tmp_path / "lr_tuning.json").read_text())["explosion_rate"]) == 2
    assert json.loads((tmp_path / "config.json").read_text())["lr"] in (0.05, 0.1)
    for args in [
        ["retrain", d, "--kind", "drop", "--alpha", "0.025", "--alpha", "0.05", "--n-subsets", "2", "--vmap-limit", "3"],  # 4 models in 2 chunks (no full model)
        ["retrain", d, "--kind", "drop", "--alpha", "0.025", "--n-subsets", "4"],  # 2 new models
        ["train", d, "--num-top-eigs", "2", "--num-neg-eigs", "1", "--eig-every", "2", "--eval-every", "4", "--n-test", "50", "--ema-decay", "0.9", "--curvature", "--no-show"],
        ["train", d, "--no-show"],  # exists: skipped
        ["magic", d, "--n-test", "5", "--n-subsets", "2", "--no-show"],  # 2 eval sets
        ["magic", d, "--n-test", "5", "--n-subsets", "3", "--no-show"],  # one new eval subset
        ["magic", d, "--n-test", "20", "--no-show"],  # another size (4 eval fns in all)
        ["magic", d, "--n-test", "5", "--no-show"],  # nothing new
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
        if args[0] == "retrain":
            assert not (tmp_path / "full_model").exists()  # retrain is independent of the full model
    results = evaluate(tmp_path, show=False, extra_step=True)
    assert [(g["kind"], g["alpha"], g["magic"], len(g["runs"])) for g in results["groups"]] == [("drop", 0.025, "magic", 4), ("drop", 0.05, "magic", 2)]
    assert [(s["n"], s["dir"]) for s in results["magic"]["magic"]["eval_sets"]] == [(5, "5_examples_0"), (5, "5_examples_1"), (5, "5_examples_2"), (20, "20_examples_0")]
    assert len(results["magic"]["magic"]["full_extra_step"]) == 4 and all(len(g["extra_step"]["metrics"]) == 4 for g in results["groups"])
    assert (tmp_path / "evaluation" / "alpha_extra_step.png").exists() and (tmp_path / "evaluation" / "drop_0.05_extra_step_pooled.png").exists()
    lin = torch.load(tmp_path / "magic" / "20_examples_0" / "linearization.pt", weights_only=True)
    assert set(lin) == {"indices", "output", "d_weights"} and len(lin["indices"]) == 20
    assert (tmp_path / "full_model" / "final.pt").exists() and not (tmp_path / "full_model" / "start.pt").exists()
    logged = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)
    assert {"loss", "acc", "margin", "params/out.weight", "grads/out.weight", "eig/1", "eig/2", "eig/-1", "eig/threshold", "eigvec/-1/out.weight",
            "test_loss", "test_acc", "loss/ema", "test_loss/ema", "ema_hess/Hv", "ema_hess/vHv", "grad_hess/vHv"} <= set(logged)
    assert len(logged["loss"]["step"]) == 12
    assert not any(k.startswith("displacement/") for k in logged)  # train logs no displacements
    assert "tangent/buffer/out.weight" not in logged and "tangent_hess/Hv" not in logged  # opt-in
    assert logged["grad_hess/vHv"]["step"].tolist() == [0, 2, 4, 6, 8, 10] and logged["ema_hess/Hv"]["step"].tolist() == [2, 4, 6, 8, 10]  # not at start
    assert len(logged["test_loss"]["step"]) == 4 and logged["eig/-1"]["step"].tolist() == [2, 4, 6, 8, 10]  # steps 0, 4, 8, 12; LOBPCG after start
    torch.testing.assert_close(logged["eig/threshold"]["value"], torch.full((5,), 2 * 1.9 / json.loads((tmp_path / "config.json").read_text())["lr"], dtype=torch.float64))  # momentum 0.9: 2 (1 + 0.9) / lr
    magic_metrics = torch.load(tmp_path / "magic" / "metrics.pt", weights_only=True)  # of the last magic call that computed something
    assert set(magic_metrics) == {"f/0", "d_state/0", "d_weights/0"}  # 1 subset: scalars only
    assert all(len(m["step"]) == 12 and m["value"].ndim == 1 for m in magic_metrics.values())  # (one per step)
    assert not (tmp_path / "retrained" / results["groups"][0]["runs"][0] / "metrics.pt").exists()  # retrained runs aren't logged
    for g in results["groups"]:
        assert g["pooled"]["ccc"] > 0.9 and g["pooled"]["spearman"] > 0.9
        if len(g["runs"]) >= 3:  # the per-set metrics of a 2-run group are degenerate (2 points)
            assert all(m["spearman"] > 0.5 for m in g["metrics"]), [round(m["spearman"], 3) for m in g["metrics"]]
        assert (tmp_path / "evaluation" / f"{g['kind']}_{g['alpha']:g}.png").exists()
    assert (tmp_path / "evaluation" / "alpha.png").exists() and (tmp_path / "evaluation" / "results.json").exists()


def test_start_and_randn(tmp_path):
    """Noising the weights only from step 6 on, evaluated at step 12: the state at step 6 is trained once (full_model/start.pt),
    the retrained models, the full model and magic start there."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--width", "16", "--loss", "mse", "--lr", "0.1", "--n-steps", "12", "--start", "6", "--dtype", "fp64"],
        ["retrain", d, "--kind", "randn", "--alpha", "0.01", "--n-subsets", "3"],
        ["train", d, "--eval-every", "0", "--tangent", "--curvature", "--eig-every", "3", "--no-show"],
        ["magic", d, "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    assert set(torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)) == {"params", "lr", "wd"}
    logged = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)
    assert "tangent/params/out.weight" in logged and logged["loss"]["step"][0] == 6 and "test_loss" not in logged
    assert logged["grad_hess/vHv"]["step"].tolist() == [6, 9] and all(torch.isfinite(logged[f"{d}_hess/{k}"]["value"]).all() for d in ("grad", "tangent", "ema") for k in ("Hv", "vHv"))
    assert logged["ema_hess/vHv"]["step"].tolist() == [9] and (logged["grad_hess/Hv"]["value"] >= logged["grad_hess/vHv"]["value"].abs()).all()  # Cauchy-Schwarz
    assert "eig/1" not in logged and torch.equal(logged["eig/threshold"]["value"], torch.full((2,), 20.0, dtype=torch.float64))  # 2 / lr
    pert = torch.load(tmp_path / "retrained" / "randn_0.01_2" / "perturbation.pt", weights_only=True)
    assert (tmp_path / "retrained" / "randn_0.01_2" / "final.pt").exists() and (pert["kind"], pert["alpha"]) == ("randn", 0.01)
    assert pert["weights"].shape == (512,) and not torch.equal(pert["weights"], torch.ones(512))
    assert set(torch.load(tmp_path / "full_model" / "start.pt", weights_only=True)) == {"params", "lr", "wd"}
    assert torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)["loss"]["step"][0] == 6  # no log before `start`
    results = evaluate(tmp_path, show=False)
    assert [(g["kind"], g["alpha"], len(g["runs"])) for g in results["groups"]] == [("randn", 0.01, 3)]
    assert all(m["ccc"] > 0.9 for m in results["groups"][0]["metrics"])
    assert not (tmp_path / "evaluation" / "alpha.png").exists()  # a single (kind, alpha)


def test_sharpness(tmp_path):
    """The sharpness optimizer (SGD with the step size 2 / top Hessian eigenvalue, a second-order step with aux state) runs
    through train and magic."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--width", "16", "--optimizer", "sharpness", "--lr", "1", "--n-steps", "6", "--dtype", "fp64"],
        ["train", d, "--eval-every", "3", "--n-test", "20", "--no-show"],
        ["magic", d, "--n-test", "5", "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    final = torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)
    assert set(final) == {"params", "lr", "wd", "eigvec", "stats"} and torch.isfinite(final["stats"]["sharpness"])
    lin = torch.load(tmp_path / "magic" / "5_examples_0" / "linearization.pt", weights_only=True)
    assert torch.isfinite(lin["d_weights"]).all() and (lin["d_weights"] != 0).any()
    full_metrics = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)
    assert "eig_iters" in full_metrics and len(full_metrics["sharpness"]["step"]) == 6 and torch.isfinite(full_metrics["sharpness"]["value"]).all()  # the stats
    logged = full_metrics
    assert "tangent/params/out.weight" not in logged and "lobpcg_iters/1" not in logged  # the step's eigenpair, no LOBPCG
    assert logged["eig/1"]["step"].tolist() == list(range(6)) and "eigvec/1/out.weight" in logged
    torch.testing.assert_close(logged["eig/1"]["value"], logged["sharpness"]["value"])  # the step's own top eigenvalue, unscaled
    torch.testing.assert_close(logged["eig/threshold"]["value"], logged["sharpness"]["value"])  # lr 1: 2 / (2 / sharpness)
    magic_metrics = torch.load(tmp_path / "magic" / "metrics.pt", weights_only=True)
    assert set(magic_metrics) == {"f/0", "d_state/0", "d_weights/0"}


def test_bf16(tmp_path):
    """fp32 states with the model forward in bfloat16 run through logging (double backward, LOBPCG), MAGIC and the evaluation."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--width", "16", "--lr", "0.1", "--n-steps", "4", "--dtype", "bf16", "--lr-schedule", "cosine", "--warmup", "2"],
        ["retrain", d, "--n-subsets", "2"],
        ["train", d, "--eval-every", "2", "--n-test", "50", "--num-top-eigs", "1", "--eig-every", "1", "--tangent", "--no-show"],
        ["magic", d, "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    logged = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)
    assert {"lr", "eig/1", "tangent/params/out.weight"} <= set(logged)
    final = torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)["params"]
    assert all(p.dtype == torch.float32 for p in final.values())
    lin = torch.load(tmp_path / "magic" / "1_examples_0" / "linearization.pt", weights_only=True)
    assert torch.isfinite(lin["d_weights"]).all() and (lin["d_weights"] != 0).any()
    results = evaluate(tmp_path, show=False)
    assert len(results["groups"][0]["runs"]) == 2


def test_centralflow(tmp_path):
    """The central flow optimizer (SUBSTEPS second-order steps per config step, with its eigenvectors as aux state and an
    implicit backward) runs through train and magic, and holds the top Hessian eigenvalue at 2 / lr."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--width", "16", "--optimizer", "centralflow", "--lr", "2", "--n-steps", "3", "--dtype", "fp64"],
        ["train", d, "--eval-every", "2", "--n-test", "20", "--no-show"],
        ["magic", d, "--n-test", "5", "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    final = torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)
    assert set(final) == {"params", "lr", "wd", "eigvecs", "curvatures", "curvature_grads", "stats", "compile", "devices"}
    assert all(v.shape == (2, *final["params"][k].shape) for k, v in final["eigvecs"].items())  # n_critical 1 + 1
    logged = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)
    n_substeps = 3 * CentralFlowState.SUBSTEPS
    assert all(len(logged[k]["step"]) == n_substeps for k in ("eig/1", "eig/2", "eig_iters", "n_critical", "oscillation"))
    assert (logged["n_critical"]["value"] == 1).all()
    torch.testing.assert_close(logged["eig/1"]["value"][4:12], torch.ones(8), rtol=1e-2, atol=0)  # on the edge (batches 1 and 2)
    assert (logged["oscillation"]["value"][:12] > 0).all()
    lin = torch.load(tmp_path / "magic" / "5_examples_0" / "linearization.pt", weights_only=True)
    assert torch.isfinite(lin["d_weights"]).all() and (lin["d_weights"] != 0).any()


def test_newton(tmp_path):
    """The newton optimizer (damped Newton by MINRES: a second-order step with aux state and an implicit backward) runs
    through train and magic."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--width", "16", "--optimizer", "newton", "--lr", "1", "--n-steps", "6", "--dtype", "fp64"],
        ["train", d, "--num-top-eigs", "1", "--eig-every", "2", "--eval-every", "3", "--n-test", "20", "--no-show"],
        ["magic", d, "--n-test", "5", "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    final = torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)
    assert set(final) == {"params", "lr", "wd", "damping", "rtol", "newton", "eigvec", "stats"}
    assert final["damping"] == 0.1 and 0 < final["stats"]["newton_iters"] < 100 and torch.isfinite(final["stats"]["residual"])  # converged
    lin = torch.load(tmp_path / "magic" / "5_examples_0" / "linearization.pt", weights_only=True)
    assert torch.isfinite(lin["d_weights"]).all() and (lin["d_weights"] != 0).any()
    logged = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)
    assert {"newton_iters", "eig_iters", "residual", "min_eig"} <= set(logged) and len(logged["min_eig"]["step"]) == 6  # the stats
    assert "tangent/params/out.weight" not in logged and torch.isfinite(logged["eig/1"]["value"]).all()
    assert logged["eig/1"]["step"].tolist() == [2, 4] and logged["eig/-1"]["step"].tolist() == list(range(6))  # LOBPCG after start; the step's own min eig
    torch.testing.assert_close(logged["eig/-1"]["value"], logged["min_eig"]["value"])  # unscaled
    torch.testing.assert_close(logged["eig/threshold"]["value"], 2 * ((-2 * logged["min_eig"]["value"]).clamp_min(0) + 0.1))  # 2 / (lr / shift), lr 1, damping 0.1
    magic_metrics = torch.load(tmp_path / "magic" / "metrics.pt", weights_only=True)
    assert set(magic_metrics) == {"f/0", "d_state/0", "d_weights/0"}


def test_prune(tmp_path):
    """`prune`: the losses saved by the evaluation give the same results"""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--width", "16", "--loss", "mse", "--lr", "0.1", "--n-steps", "8", "--start", "2", "--dtype", "fp64"],
        ["retrain", d, "--kind", "randn", "--alpha", "0.01", "--n-subsets", "3"],
        ["train", d, "--eval-every", "0", "--no-show"],
        ["magic", d, "--n-test", "5", "--n-subsets", "2", "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    exp = Experiment(tmp_path, "cpu", False)
    lin = torch.load(tmp_path / "magic" / "5_examples_1" / "linearization.pt", weights_only=True)
    mask = torch.zeros(len(exp.train_ds), dtype=torch.bool)  # the examples of a batch of the steps 2-7
    mask[torch.cat([exp.loader.get_batch(t)[2] for t in range(2, 8)])] = True
    assert (lin["d_weights"][~mask] == 0).all() and (lin["d_weights"][mask] != 0).all()
    results_all = evaluate(tmp_path, show=False)
    assert all(g["pooled"]["ccc"] > 0.9 for g in results_all["groups"]), [g["pooled"] for g in results_all["groups"]]

    # the states go, the losses saved by the evaluation (losses.json) give the same evaluation
    result = runner.invoke(main, ["prune", d, "-y"], catch_exceptions=False)
    assert result.exit_code == 0 and "deleted 5 files" in result.output, result.output
    assert not list(tmp_path.glob("*/*/final.pt")) and not (tmp_path / "full_model" / "final.pt").exists()
    assert set(json.loads((tmp_path / "retrained" / "randn_0.01_0" / "losses.json").read_text())) == {"final"}
    pruned = evaluate(tmp_path, show=False)
    assert pruned["groups"][0]["reals"] == results_all["groups"][0]["reals"] and pruned["magic"]["magic"]["full"] == results_all["magic"]["magic"]["full"]
    with pytest.raises(click.UsageError, match="pruned"):  # an eval set never evaluated needs the states
        exp.eval_losses(tmp_path / "full_model", [exp.test_ds.subset(3, index=0)])


def test_checkpoints_and_resume(tmp_path, monkeypatch):
    """`train --checkpoint-every` saves the states magic backpropagates between, and a train / magic killed after a
    checkpoint continues from it: the same final state, logs and df/dw as the plain run."""
    from src.data import Loader
    runner = CliRunner()
    new = ["--dataset", "majority", "--model", "mlp", "--width", "16", "--optimizer", "momentum", "--lr", "0.1", "--n-steps", "8", "--dtype", "fp64"]
    train = ["--num-top-eigs", "2", "--eig-every", "2", "--eval-every", "4", "--n-test", "20", "--tangent", "--no-show"]
    plain, killed = tmp_path / "plain", tmp_path / "killed"
    for args in [["new", str(plain), *new], ["train", str(plain), *train], ["magic", str(plain), "--n-test", "5", "--n-subsets", "2", "--no-show"],
                 ["new", str(killed), *new]]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output

    get_batch = Loader.get_batch
    def kill_at(step):
        def killed_get_batch(self, t):
            if t == step:
                raise RuntimeError("killed")
            return get_batch(self, t)
        monkeypatch.setattr(Loader, "get_batch", killed_get_batch)

    kill_at(7)
    with pytest.raises(RuntimeError, match="killed"):
        runner.invoke(main, ["train", str(killed), *train, "--checkpoint-every", "3"], catch_exceptions=False)
    assert sorted(p.name for p in (killed / "full_model" / "checkpoints").iterdir()) == ["3.pt", "6.pt"]
    assert torch.load(killed / "full_model" / "resume.pt", weights_only=True)["step"] == 6 and not (killed / "full_model" / "final.pt").exists()
    partial = torch.load(killed / "full_model" / "metrics.pt", weights_only=True)  # the logs so far, plotted
    assert partial["loss"]["step"].max() < 6 and (killed / "full_model" / "metrics.png").exists()
    with pytest.raises(click.UsageError, match="train"):
        runner.invoke(main, ["magic", str(killed), "--no-show"], catch_exceptions=False, standalone_mode=False)
    monkeypatch.setattr(Loader, "get_batch", get_batch)
    result = runner.invoke(main, ["train", str(killed), *train, "--checkpoint-every", "3"], catch_exceptions=False)
    assert result.exit_code == 0 and "continuing" in result.output, result.output
    assert not (killed / "full_model" / "resume.pt").exists()
    final, ref = (torch.load(d / "full_model" / "final.pt", weights_only=True) for d in (killed, plain))
    torch.testing.assert_close((final["params"], final["buffer"]), (ref["params"], ref["buffer"]), rtol=0, atol=0)
    logged, ref = (torch.load(d / "full_model" / "metrics.pt", weights_only=True) for d in (killed, plain))
    assert {"eig/1", "test_loss/ema", "tangent/params/out.weight"} <= set(logged)
    torch.testing.assert_close(logged, ref, rtol=0, atol=0)

    magic_args = ["magic", str(killed), "--n-test", "5", "--n-subsets", "2", "--passes", "1", "--no-show"]
    kill_at(1)  # in the last segment (steps 0-2)
    with pytest.raises(RuntimeError, match="killed"):
        runner.invoke(main, magic_args, catch_exceptions=False)
    assert torch.load(killed / "magic" / "resume.pt", weights_only=True)["step"] == 3
    assert (killed / "magic" / "metrics.pt").exists() and (killed / "magic" / "metrics.png").exists()
    monkeypatch.setattr(Loader, "get_batch", get_batch)
    result = runner.invoke(main, magic_args, catch_exceptions=False)
    assert result.exit_code == 0 and "continuing from step 3" in result.output, result.output
    assert not (killed / "magic" / "resume.pt").exists()
    for i in range(2):
        lin, ref = (torch.load(d / "magic" / f"5_examples_{i}" / "linearization.pt", weights_only=True) for d in (killed, plain))
        torch.testing.assert_close(lin, ref)
    logged, ref = (torch.load(d / "magic" / "metrics.pt", weights_only=True) for d in (killed, plain))
    assert logged["f/0"]["step"].tolist() == ref["f/0"]["step"].tolist() == list(range(7, -1, -1))
    torch.testing.assert_close(logged["d_weights/1"]["value"], ref["d_weights/1"]["value"])


def test_seeds(tmp_path):
    """n_seeds 2 trains the models of init seeds 3 and 4 vmapped: each seed's logs, final state and MAGIC df/dw are those of its own single run"""
    runner = CliRunner()
    config = ["--dataset", "majority", "--model", "mlp", "--width", "16", "--n-steps", "6", "--lr", "0.1", "--dtype", "fp64"]
    for name, seeds in (("both", ["--init-seed", "3", "--n-seeds", "2"]), ("seed3", ["--init-seed", "3"]), ("seed4", ["--init-seed", "4"])):
        for args in (["new", str(tmp_path / name), *config, *seeds], ["train", str(tmp_path / name), "--eval-every", "2", "--n-test", "0", "--no-show"]):
            result = runner.invoke(main, args, catch_exceptions=False)
            assert result.exit_code == 0, result.output
    both = torch.load(tmp_path / "both" / "full_model" / "metrics.pt", weights_only=True)
    final = torch.load(tmp_path / "both" / "full_model" / "final.pt", weights_only=True)
    for r, name in enumerate(("seed3", "seed4")):
        single = torch.load(tmp_path / name / "full_model" / "metrics.pt", weights_only=True)
        assert set(single) == set(both)
        for k in single:
            torch.testing.assert_close(both[k]["value"][:, r], single[k]["value"], msg=k)
        single_final = torch.load(tmp_path / name / "full_model" / "final.pt", weights_only=True)
        for k, p in single_final["params"].items():
            torch.testing.assert_close(final["params"][k][r], p)
    for name in ("both", "seed3", "seed4"):
        result = runner.invoke(main, ["magic", str(tmp_path / name), "--n-test", "3", "--no-show"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
    both_lin = torch.load(tmp_path / "both" / "magic" / "3_examples_0" / "linearization.pt", weights_only=True)
    for r, name in enumerate(("seed3", "seed4")):
        single_lin = torch.load(tmp_path / name / "magic" / "3_examples_0" / "linearization.pt", weights_only=True)
        torch.testing.assert_close(both_lin["output"][r], single_lin["output"])
        torch.testing.assert_close(both_lin["d_weights"][r], single_lin["d_weights"])
    with pytest.raises(click.UsageError):
        main.main(["retrain", str(tmp_path / "both")], standalone_mode=False)


def test_extend(tmp_path):
    """A 2-seed run of 6 steps extended to 10 (--extend-to, twice: the second is a no-op) logs what a 10-step run does, but the
    EMA of the params, restarted at step 6, and grad_displacement at step 6 (no previous gradients)"""
    runner = CliRunner()
    config = ["--dataset", "majority", "--model", "mlp", "--width", "16", "--lr", "0.1", "--dtype", "fp64", "--n-seeds", "2"]
    train = ["--eval-every", "2", "--n-test", "0", "--ema-gamma", "0", "--checkpoint-every", "4", "--no-show"]
    for args in (["new", str(tmp_path / "ext"), *config, "--n-steps", "6"], ["train", str(tmp_path / "ext"), *train],
                 ["train", str(tmp_path / "ext"), *train, "--extend-to", "10"], ["train", str(tmp_path / "ext"), *train, "--extend-to", "10"],
                 ["new", str(tmp_path / "ref"), *config, "--n-steps", "10"], ["train", str(tmp_path / "ref"), *train]):
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "ext" / "config.json").read_text())["n_steps"] == 10
    ext, ref = (torch.load(tmp_path / name / "full_model" / "metrics.pt", weights_only=True) for name in ("ext", "ref"))
    assert set(ext) == set(ref)
    ref["grad_displacement"] = {name: v[ref["grad_displacement"]["step"] != 6] for name, v in ref["grad_displacement"].items()}
    for k in ref:
        assert ext[k]["step"].tolist() == ref[k]["step"].tolist(), k
        if k.endswith("/ema_short"):  # the same until the restart
            torch.testing.assert_close(ext[k]["value"][ext[k]["step"] < 6], ref[k]["value"][ref[k]["step"] < 6])
        else:
            torch.testing.assert_close(ext[k]["value"], ref[k]["value"], msg=k)
    for name in ("ext", "ref"):
        assert not (tmp_path / name / "full_model" / "resume.pt").exists()
    final_ext, final_ref = (torch.load(tmp_path / name / "full_model" / "final.pt", weights_only=True) for name in ("ext", "ref"))
    for k, p in final_ref["params"].items():
        torch.testing.assert_close(final_ext["params"][k], p)
