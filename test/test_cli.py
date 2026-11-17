import json
import click
import pytest

from click.testing import CliRunner
import torch

from experiments.evaluate import evaluate
from experiments.fit import fit
from experiments.magic_eigs import magic_eigs
from experiments.tune_lr import tune_lr
from main import Experiment, main
from src.optimizers import CentralFlowState


def test_cli_end_to_end(tmp_path):
    """new -> tune_lr -> retrain (twice) -> train -> magic -> evaluate on a tiny CPU config, with momentum.
    MAGIC's predictions should agree with the retrained models (in float64: the loss changes are tiny)."""
    d = str(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["new", d, "--dataset", "majority", "--model", "mlp", "--n-steps", "12", "--optimizer", "momentum", "--dtype", "fp64"], catch_exceptions=False)
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
    out = magic_eigs(tmp_path, n_test=5, n_subsets=2, kind="drop", alpha=0.025, index=0, n_eigs=2, n_neg_eigs=1, eig_every=4, show=False)
    eigs = torch.load(out / "metrics.pt", weights_only=True)
    assert out == tmp_path / "magic_eigs" / "zero_drop_0.025_0"
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert eigs["eig/1"]["step"].tolist() == [0, 4, 8] and eigs["tangent_eig/-1"]["step"].tolist() == [0, 4, 8]  # |cos| of the tangent with the eigenvectors
    assert (eigs["tangent_eig/1"]["value"][1:] <= 1 + 1e-6).all() and (eigs["eig/1"]["value"] >= eigs["eig/2"]["value"]).all()
    assert eigs["eig/tangent_curvature"]["step"].tolist() == list(range(12)) and eigs["magic/0"]["step"].tolist() == list(range(13))  # every step, and the final state
    assert {"tangent/params", "tangent/buffer", "displacement/tangent/buffer", "growth/tangent", "tangent_cos/step", "magic/batch", "magic_cos/1", "magic_examples/1", "grad_f/1"} <= set(eigs)
    assert "growth/sgd_linear" not in eigs and "tangent_cos/start" not in eigs  # momentum; zero initial tangent
    dw = Experiment(tmp_path).train_ds.weights("drop", 0.025, 0) - 1
    for i in range(2):  # at the last step, df/ds_T . v_T is MAGIC's prediction df/dw . dw
        beta = torch.load(tmp_path / "magic" / f"5_examples_{i}" / "linearization.pt", weights_only=True)["beta"]
        torch.testing.assert_close(eigs[f"magic/{i}"]["value"][-1], beta @ dw)
    tangent = torch.load(out / "tangent.pt", weights_only=True)
    assert set(tangent["tangent"]) == {"params", "buffer"} and tangent["kind"] == "drop" and tangent["perturb_steps"] is None

    assert [(g["kind"], g["alpha"], g["perturb_steps"], g["magic"], len(g["runs"])) for g in results["groups"]] == [("drop", 0.025, None, "magic", 4), ("drop", 0.05, None, "magic", 2)]
    assert [(s["n"], s["dir"]) for s in results["magic"]["magic"]["eval_sets"]] == [(5, "5_examples_0"), (5, "5_examples_1"), (5, "5_examples_2"), (20, "20_examples_0")]
    assert len(results["magic"]["magic"]["full_extra_step"]) == 4 and all(len(g["extra_step"]["metrics"]) == 4 for g in results["groups"])
    assert (tmp_path / "evaluation" / "alpha_extra_step.png").exists() and (tmp_path / "evaluation" / "drop_0.05_extra_step_pooled.png").exists()
    lin = torch.load(tmp_path / "magic" / "20_examples_0" / "linearization.pt", weights_only=True)
    assert set(lin) == {"indices", "output", "beta"} and len(lin["indices"]) == 20
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
    assert "delta/0/buffer/out.weight" in magic_metrics and len(magic_metrics) == 2 * 6 + 1 + 2 + 1  # 1 subset: (params, buffer) x 6 tensors + d_weights + 2 displacements + f
    assert len(magic_metrics["displacement/delta/params/0"]["step"]) == 11 and len(magic_metrics["f/0"]["step"]) == 12  # 12 logged deltas (one per step)
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
        ["new", d, "--dataset", "majority", "--model", "mlp", "--loss", "mse", "--lr", "0.1", "--n-steps", "12", "--start", "6", "--dtype", "fp64"],
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
    out = magic_eigs(tmp_path, n_test=1, kind="randn", alpha=0.01, index=2, n_eigs=1, eig_every=3, show=False)
    eigs = torch.load(out / "metrics.pt", weights_only=True)
    assert out == tmp_path / "magic_eigs" / "zero_randn_0.01_2" and eigs["eig/1"]["step"].tolist() == [6, 9]
    beta = torch.load(tmp_path / "magic" / "1_examples_0" / "linearization.pt", weights_only=True)["beta"]
    torch.testing.assert_close(eigs["magic/0"]["value"][-1], beta @ (pert["weights"] - 1))  # the retrained model's perturbation #2
    out = magic_eigs(tmp_path, n_test=1, init="eig", init_norm=2.0, perturb_steps=0, show=False)  # the top eigenvector at step 6, no perturbation
    eigs = torch.load(out / "metrics.pt", weights_only=True)
    assert out == tmp_path / "magic_eigs" / "eig_none" and eigs["tangent_cos/start"]["step"].tolist() == list(range(6, 13))
    torch.testing.assert_close(eigs["tangent/params"]["value"][0], torch.tensor(2.0, dtype=torch.float64))
    torch.testing.assert_close(eigs["tangent_cos/start"]["value"][0], torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(eigs["growth/sgd_linear"]["value"], eigs["growth/tangent"]["value"])  # (I - lr H) v is the whole tangent step
    torch.testing.assert_close(eigs["growth/tangent"]["value"][0].abs(), (1 - 0.1 * eigs["eig/tangent_curvature"]["value"][0]).abs(), rtol=0.05, atol=0)  # an eigenvector
    assert torch.isfinite(eigs["magic/0"]["value"]).all() and (eigs["magic/0"]["value"] != 0).all()
    assert [(g["kind"], g["alpha"], g["perturb_steps"], len(g["runs"])) for g in results["groups"]] == [("randn", 0.01, None, 3)]
    assert all(m["ccc"] > 0.9 for m in results["groups"][0]["metrics"])
    assert not (tmp_path / "evaluation" / "alpha.png").exists()  # a single (kind, alpha)


def test_sharpness(tmp_path):
    """The sharpness optimizer (SGD with the step size 2 / top Hessian eigenvalue, a second-order step with aux state) runs
    through train and magic."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--optimizer", "sharpness", "--lr", "1", "--n-steps", "6", "--dtype", "fp64"],
        ["train", d, "--eval-every", "3", "--n-test", "20", "--no-show"],
        ["magic", d, "--n-test", "5", "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    final = torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)
    assert set(final) == {"params", "lr", "wd", "eigvec", "stats"} and torch.isfinite(final["stats"]["sharpness"])
    lin = torch.load(tmp_path / "magic" / "5_examples_0" / "linearization.pt", weights_only=True)
    assert torch.isfinite(lin["beta"]).all() and (lin["beta"] != 0).any()
    full_metrics = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)
    assert "eig_iters" in full_metrics and len(full_metrics["sharpness"]["step"]) == 6 and torch.isfinite(full_metrics["sharpness"]["value"]).all()  # the stats
    logged = full_metrics
    assert "tangent/params/out.weight" not in logged and "lobpcg_iters/1" not in logged  # the step's eigenpair, no LOBPCG
    assert logged["eig/1"]["step"].tolist() == list(range(6)) and "eigvec/1/out.weight" in logged
    torch.testing.assert_close(logged["eig/1"]["value"], logged["sharpness"]["value"])  # the step's own top eigenvalue, unscaled
    torch.testing.assert_close(logged["eig/threshold"]["value"], logged["sharpness"]["value"])  # lr 1: 2 / (2 / sharpness)
    magic_metrics = torch.load(tmp_path / "magic" / "metrics.pt", weights_only=True)
    assert not any(k.startswith("delta/5 examples #0/eigvec") for k in magic_metrics)  # no norms of the aux groups


def test_bf16(tmp_path):
    """fp32 states with the model forward in bfloat16 run through logging (double backward, LOBPCG), MAGIC and the evaluation."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--lr", "0.1", "--n-steps", "4", "--dtype", "bf16", "--lr-schedule", "cosine", "--warmup", "2"],
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
    assert torch.isfinite(lin["beta"]).all() and (lin["beta"] != 0).any()
    results = evaluate(tmp_path, show=False)
    out = magic_eigs(tmp_path, n_test=5, n_subsets=2, n_eigs=1, eig_every=2, show=False)
    eigs = torch.load(out / "metrics.pt", weights_only=True)
    assert eigs["eig/1"]["step"].tolist() == [0, 2] and eigs["magic/1"]["step"].tolist() == [0, 1, 2, 3, 4] and torch.isfinite(eigs["magic/1"]["value"]).all()
    assert len(results["groups"][0]["runs"]) == 2


def test_centralflow(tmp_path):
    """The central flow optimizer (SUBSTEPS second-order steps per config step, with its eigenvectors as aux state and an
    implicit backward) runs through train and magic, and holds the top Hessian eigenvalue at 2 / lr."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--width", "16", "--optimizer", "centralflow", "--lr", "2", "--n-steps", "4", "--dtype", "fp64"],
        ["train", d, "--eval-every", "2", "--n-test", "20", "--no-show"],
        ["magic", d, "--n-test", "5", "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    final = torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)
    assert set(final) == {"params", "lr", "wd", "eigvecs", "stats", "compile", "devices"}
    assert all(v.shape == (4, *final["params"][k].shape) for k, v in final["eigvecs"].items())
    logged = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)
    n_substeps = 4 * CentralFlowState.SUBSTEPS
    assert all(len(logged[k]["step"]) == n_substeps for k in ("eig/1", "eig/4", "eig/threshold", "eig_iters", "n_critical", "oscillation"))
    assert (logged["eig/threshold"]["value"] == 1).all() and (logged["n_critical"]["value"] == 1).all()
    torch.testing.assert_close(logged["eig/1"]["value"][4:12], torch.ones(8), rtol=1e-2, atol=0)  # on the edge (batches 1 and 2)
    assert (logged["oscillation"]["value"][:12] > 0).all()
    lin = torch.load(tmp_path / "magic" / "5_examples_0" / "linearization.pt", weights_only=True)
    assert torch.isfinite(lin["beta"]).all() and (lin["beta"] != 0).any()


def test_newton(tmp_path):
    """The newton optimizer (damped Newton by MINRES: a second-order step with aux state and an implicit backward) runs
    through train and magic."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--optimizer", "newton", "--lr", "1", "--n-steps", "6", "--dtype", "fp64"],
        ["train", d, "--num-top-eigs", "1", "--eig-every", "2", "--eval-every", "3", "--n-test", "20", "--no-show"],
        ["magic", d, "--n-test", "5", "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    final = torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)
    assert set(final) == {"params", "lr", "wd", "damping", "rtol", "newton", "eigvec", "stats"}
    assert final["damping"] == 0.1 and 0 < final["stats"]["newton_iters"] < 100 and torch.isfinite(final["stats"]["residual"])  # converged
    lin = torch.load(tmp_path / "magic" / "5_examples_0" / "linearization.pt", weights_only=True)
    assert torch.isfinite(lin["beta"]).all() and (lin["beta"] != 0).any()
    logged = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)
    assert {"newton_iters", "eig_iters", "residual", "min_eig"} <= set(logged) and len(logged["min_eig"]["step"]) == 6  # the stats
    assert "tangent/params/out.weight" not in logged and torch.isfinite(logged["eig/1"]["value"]).all()
    assert logged["eig/1"]["step"].tolist() == [2, 4] and logged["eig/-1"]["step"].tolist() == list(range(6))  # LOBPCG after start; the step's own min eig
    torch.testing.assert_close(logged["eig/-1"]["value"], logged["min_eig"]["value"])  # unscaled
    torch.testing.assert_close(logged["eig/threshold"]["value"], 2 * ((-2 * logged["min_eig"]["value"]).clamp_min(0) + 0.1))  # 2 / (lr / shift), lr 1, damping 0.1
    magic_metrics = torch.load(tmp_path / "magic" / "metrics.pt", weights_only=True)
    assert not any(k.startswith("delta/5 examples #0/newton") for k in magic_metrics)  # no norms of the aux groups


def test_perturb_steps(tmp_path):
    """--perturb-steps k: retrain noises the weights of the first k steps from `start` alone (the model matches a manual k-step
    perturbed train, differs from the all-steps one, is refused for --kind drop), magic adds df/dw_t for those steps alone
    (DIR/magic_steps<k>), and evaluate compares each group with the magic dir of its perturb_steps, in one evaluation."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--loss", "mse", "--lr", "0.1", "--n-steps", "8", "--start", "2", "--dtype", "fp64"],
        ["retrain", d, "--kind", "randn", "--alpha", "0.01", "--n-subsets", "3", "--perturb-steps", "2"],
        ["retrain", d, "--kind", "randn", "--alpha", "0.01", "--n-subsets", "3"],
        ["train", d, "--eval-every", "0", "--no-show"],
        ["magic", d, "--n-test", "5", "--n-subsets", "2", "--perturb-steps", "2", "--no-show"],
        ["magic", d, "--n-test", "5", "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    for args in (["retrain", d, "--kind", "drop", "--perturb-steps", "1"], ["magic", d, "--perturb-steps", "7"]):
        result = runner.invoke(main, args)
        assert result.exit_code != 0 and ("only allowed with --kind randn" in result.output or "must be in 1 ... 6" in result.output), result.output
    pert = torch.load(tmp_path / "retrained" / "steps2_randn_0.01_1" / "perturbation.pt", weights_only=True)
    assert pert["perturb_steps"] == 2 and (pert["kind"], pert["alpha"]) == ("randn", 0.01)
    assert torch.load(tmp_path / "retrained" / "randn_0.01_1" / "perturbation.pt", weights_only=True)["perturb_steps"] is None
    exp = Experiment(tmp_path, "cpu", False)
    expected = exp.magic.train(exp.magic.train(exp.start_state, n_steps=2, start=2, weights=pert["weights"]), n_steps=4, start=4)
    got = exp.load_state(tmp_path / "retrained" / "steps2_randn_0.01_1" / "final.pt")
    torch.testing.assert_close(got.params, expected.params)
    assert not torch.allclose(got.params["out.weight"], exp.load_state(tmp_path / "retrained" / "randn_0.01_1" / "final.pt").params["out.weight"])
    # magic --perturb-steps 2: df/dw = 0 for the examples of no perturbed batch
    lin = torch.load(tmp_path / "magic_steps2" / "5_examples_1" / "linearization.pt", weights_only=True)
    perturbed = torch.cat([exp.loader.get_batch(t)[2] for t in (2, 3)])
    mask = torch.zeros(len(exp.train_ds), dtype=torch.bool)
    mask[perturbed] = True
    assert (lin["beta"][~mask] == 0).all() and (lin["beta"][mask] != 0).all()
    lin0, lin0_all = (torch.load(tmp_path / m / "5_examples_0" / "linearization.pt", weights_only=True) for m in ("magic_steps2", "magic"))
    torch.testing.assert_close(lin0["output"], lin0_all["output"])  # the same training, so the same f
    assert not torch.allclose(lin0["beta"], lin0_all["beta"])  # the later steps' df/dw_t are left out

    results = results_all = evaluate(tmp_path, show=False)
    assert [(g["kind"], g["alpha"], g["perturb_steps"], g["magic"], len(g["runs"])) for g in results["groups"]] == [("randn", 0.01, None, "magic", 3), ("randn", 0.01, 2, "magic_steps2", 3)]
    assert results["groups"][1]["runs"] == ["steps2_randn_0.01_0", "steps2_randn_0.01_1", "steps2_randn_0.01_2"]
    assert [s["dir"] for s in results["magic"]["magic_steps2"]["eval_sets"]] == ["5_examples_0", "5_examples_1"]
    assert all(g["pooled"]["ccc"] > 0.9 for g in results["groups"]), [g["pooled"] for g in results["groups"]]
    assert (tmp_path / "evaluation" / "randn_0.01_steps2_pooled.png").exists() and (tmp_path / "evaluation" / "alpha.png").exists()

    # the linear-regression baseline on the 3 steps2 models (fewer models than perturbed examples: the minimum-norm fit)
    out = fit(tmp_path, n_test=5, n_subsets=2, perturb_steps=2, folds=3, show=False)
    lin_fit = torch.load(out / "5_examples_1" / "linearization.pt", weights_only=True)
    assert out == tmp_path / "fit_steps2" and (lin_fit["beta"][~mask] == 0).all() and torch.isfinite(lin_fit["beta"]).all()
    torch.testing.assert_close(lin_fit["output"], lin["output"])
    assert torch.equal(lin_fit["indices"], lin["indices"])
    assert set(lin_fit["cv_preds"]) == {"steps2_randn_0.01_0", "steps2_randn_0.01_1", "steps2_randn_0.01_2"}
    results = evaluate(tmp_path, show=False, method="fit")  # (the all-steps models have no fit dir: skipped)
    assert [(g["perturb_steps"], g["magic"], len(g["runs"])) for g in results["groups"]] == [(2, "fit_steps2", 3)]
    assert (tmp_path / "evaluation_fit" / "randn_0.01_steps2_pooled.png").exists()

    # prune: the states go, the losses saved by the evaluations (losses.json) give the same evaluation and fit
    result = runner.invoke(main, ["prune", d, "-y"], catch_exceptions=False)
    assert result.exit_code == 0 and "deleted 8 files" in result.output, result.output
    assert not list(tmp_path.glob("*/*/final.pt")) and not (tmp_path / "full_model" / "final.pt").exists()
    assert set(json.loads((tmp_path / "retrained" / "randn_0.01_0" / "losses.json").read_text())) == {"final"}
    pruned = evaluate(tmp_path, show=False)
    assert pruned["groups"][0]["reals"] == results_all["groups"][0]["reals"] and pruned["magic"]["magic"]["full"] == results_all["magic"]["magic"]["full"]
    assert torch.equal(torch.load(fit(tmp_path, n_test=5, n_subsets=2, perturb_steps=2, folds=0, show=False) / "5_examples_1" / "linearization.pt", weights_only=True)["beta"], lin_fit["beta"])
    with pytest.raises(click.UsageError, match="pruned"):  # an eval set never evaluated needs the states
        fit(tmp_path, n_test=3, n_subsets=1, perturb_steps=2, show=False)


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
