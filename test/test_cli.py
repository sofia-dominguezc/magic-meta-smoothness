import json

from click.testing import CliRunner
import torch

from experiments.evaluate import evaluate
from experiments.magic_eigs import magic_eigs
from experiments.tune_lr import tune_lr
from main import main


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
        ["train", d, "--num-top-eigs", "2", "--num-neg-eigs", "1", "--eig-every", "2", "--eval-every", "4", "--n-test", "50", "--no-show"],
        ["train", d, "--no-show"],  # exists: skipped
        ["magic", d, "--n-test", "5", "--n-subsets", "2", "--vmap-limit", "1", "--no-show"],  # 2 eval sets in 2 chunks
        ["magic", d, "--n-test", "5", "--n-subsets", "3", "--no-show"],  # one new eval subset
        ["magic", d, "--n-test", "20", "--no-show"],  # another size (4 eval fns in all)
        ["magic", d, "--n-test", "5", "--no-show"],  # nothing new
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
        if args[0] == "retrain":
            assert not (tmp_path / "full_model").exists()  # retrain is independent of the full model
    results = evaluate(tmp_path, show=False)
    out = magic_eigs(tmp_path, n_test=5, n_subsets=2, n_eigs=2, n_neg_eigs=1, eig_every=4, show=False)
    eigs = torch.load(out / "metrics.pt", weights_only=True)
    cfg = json.loads((tmp_path / "config.json").read_text())
    eig_steps = [t for t in reversed(range(cfg["start"], cfg["n_steps"])) if (t - cfg["start"]) % 4 == 0]
    assert eigs["eig/1"]["step"].tolist() == eig_steps and eigs["delta_eig/-1"]["value"].shape == (len(eig_steps), 2)  # a |cos| per eval set
    assert (eigs["delta_eig/1"]["value"] <= 1 + 1e-6).all() and (eigs["eig/1"]["value"] >= eigs["eig/2"]["value"]).all()
    assert (eigs["eig/gn1"]["value"] >= eigs["eig/gn2"]["value"]).all() and eigs["delta_eig/gn2"]["value"].shape == (len(eig_steps), 2)
    assert "displacement/delta/params/0" in eigs and "displacement/delta/params/1" in eigs  # total norms: a panel per group, a line per eval set

    assert [(g["kind"], g["alpha"], len(g["runs"])) for g in results["groups"]] == [("drop", 0.025, 4), ("drop", 0.05, 2)]
    assert [(s["n"], s["dir"]) for s in results["eval_sets"]] == [(5, "5_examples_0"), (5, "5_examples_1"), (5, "5_examples_2"), (20, "20_examples_0")]
    lin = torch.load(tmp_path / "magic" / "20_examples_0" / "linearization.pt", weights_only=True)
    assert set(lin) == {"indices", "output", "beta"} and len(lin["indices"]) == 20
    assert (tmp_path / "full_model" / "final.pt").exists() and not (tmp_path / "full_model" / "start.pt").exists()
    logged = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)
    assert {"loss", "acc", "margin", "params/out.weight", "grads/out.weight", "eig/1", "eig/2", "eig/-1", "eig/threshold", "eigvec/-1/out.weight",
            "ema_eig/1", "loss/ema", "test_loss", "test_loss/ema"} <= set(logged)
    assert len(logged["loss"]["step"]) == 12
    assert not any(k.startswith("displacement/") for k in logged)  # train logs no displacements
    assert "tangent/buffer/out.weight" not in logged and "eig/batch_sharpness" not in logged  # opt-in
    assert len(logged["test_loss"]["step"]) == 4 and logged["eig/-1"]["step"].tolist() == [4, 6, 8, 10]  # steps 0, 4, 8, 12; LOBPCG from start + 2 eig_every
    torch.testing.assert_close(logged["eig/threshold"]["value"], torch.full((4,), 2 * 1.9 / json.loads((tmp_path / "config.json").read_text())["lr"], dtype=torch.float64))  # momentum 0.9: 2 (1 + 0.9) / lr
    magic_metrics = torch.load(tmp_path / "magic" / "metrics.pt", weights_only=True)  # of the last magic call that computed something
    assert "delta/0/buffer/out.weight" in magic_metrics and len(magic_metrics) == 2 * 6 + 1 + 2 + 1  # 1 subset: (params, buffer) x 6 tensors + d_weights + 2 displacements + f
    assert len(magic_metrics["displacement/delta/params/0"]["step"]) == 12 and len(magic_metrics["f/0"]["step"]) == 13  # 13 logged deltas (df/ds_T and the 12 steps)
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
        ["train", d, "--eval-every", "0", "--tangent", "--batch-sharpness", "--eig-every", "3", "--no-show"],
        ["magic", d, "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    assert set(torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)) == {"params", "lr", "wd", "ema_decay", "ema_params"}
    logged = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)
    assert "tangent/params/out.weight" in logged and logged["loss"]["step"][0] == 6 and "test_loss" not in logged
    assert logged["eig/batch_sharpness"]["step"].tolist() == [6, 9] and torch.isfinite(logged["eig/batch_sharpness"]["value"]).all()
    assert "eig/1" not in logged and torch.equal(logged["eig/threshold"]["value"], torch.full((2,), 20.0, dtype=torch.float64))  # 2 / lr
    pert = torch.load(tmp_path / "retrained" / "randn_0.01_2" / "perturbation.pt", weights_only=True)
    assert (tmp_path / "retrained" / "randn_0.01_2" / "final.pt").exists() and (pert["kind"], pert["alpha"]) == ("randn", 0.01)
    assert pert["weights"].shape == (512,) and not torch.equal(pert["weights"], torch.ones(512))
    assert set(torch.load(tmp_path / "full_model" / "start.pt", weights_only=True)) == {"params", "lr", "wd", "ema_decay", "ema_params"}
    assert torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)["loss"]["step"][0] == 6  # no log before `start`
    results = evaluate(tmp_path, show=False)
    out = magic_eigs(tmp_path, n_test=5, n_subsets=2, n_eigs=2, n_neg_eigs=1, eig_every=4, show=False)
    eigs = torch.load(out / "metrics.pt", weights_only=True)
    cfg = json.loads((tmp_path / "config.json").read_text())
    eig_steps = [t for t in reversed(range(cfg["start"], cfg["n_steps"])) if (t - cfg["start"]) % 4 == 0]
    assert eigs["eig/1"]["step"].tolist() == eig_steps and eigs["delta_eig/-1"]["value"].shape == (len(eig_steps), 2)  # a |cos| per eval set
    assert (eigs["delta_eig/1"]["value"] <= 1 + 1e-6).all() and (eigs["eig/1"]["value"] >= eigs["eig/2"]["value"]).all()
    assert (eigs["eig/gn1"]["value"] >= eigs["eig/gn2"]["value"]).all() and eigs["delta_eig/gn2"]["value"].shape == (len(eig_steps), 2)
    assert "displacement/delta/params/0" in eigs and "displacement/delta/params/1" in eigs  # total norms: a panel per group, a line per eval set
    assert [(g["kind"], g["alpha"], len(g["runs"])) for g in results["groups"]] == [("randn", 0.01, 3)]
    assert all(m["ccc"] > 0.9 for m in results["groups"][0]["metrics"])
    assert not (tmp_path / "evaluation" / "alpha.png").exists()  # a single (kind, alpha)


def test_sharpness(tmp_path):
    """The sharpness optimizer (SGD with the step size 2 / top Hessian eigenvalue, a second-order step with aux state) runs
    through train (not vmapped: no retrain) and magic."""
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
    assert set(final) == {"params", "lr", "wd", "ema_decay", "ema_params", "eigvec", "stats"} and torch.isfinite(final["stats"]["sharpness"])
    lin = torch.load(tmp_path / "magic" / "5_examples_0" / "linearization.pt", weights_only=True)
    assert torch.isfinite(lin["beta"]).all() and (lin["beta"] != 0).any()
    full_metrics = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)
    assert "eig_iters" in full_metrics and len(full_metrics["sharpness"]["step"]) == 6 and torch.isfinite(full_metrics["sharpness"]["value"]).all()  # the stats
    logged = full_metrics
    assert "tangent/params/out.weight" not in logged and "lobpcg_iters/1" not in logged  # the step's eigenpair, no LOBPCG
    assert logged["eig/1"]["step"].tolist() == list(range(6)) and "eigvec/1/out.weight" in logged and "ema_eig/1" in logged
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
    lin = torch.load(tmp_path / "magic" / "10_examples_0" / "linearization.pt", weights_only=True)
    assert torch.isfinite(lin["beta"]).all() and (lin["beta"] != 0).any()
    results = evaluate(tmp_path, show=False)
    out = magic_eigs(tmp_path, n_test=5, n_subsets=2, n_eigs=2, n_neg_eigs=1, eig_every=4, show=False)
    eigs = torch.load(out / "metrics.pt", weights_only=True)
    cfg = json.loads((tmp_path / "config.json").read_text())
    eig_steps = [t for t in reversed(range(cfg["start"], cfg["n_steps"])) if (t - cfg["start"]) % 4 == 0]
    assert eigs["eig/1"]["step"].tolist() == eig_steps and eigs["delta_eig/-1"]["value"].shape == (len(eig_steps), 2)  # a |cos| per eval set
    assert (eigs["delta_eig/1"]["value"] <= 1 + 1e-6).all() and (eigs["eig/1"]["value"] >= eigs["eig/2"]["value"]).all()
    assert (eigs["eig/gn1"]["value"] >= eigs["eig/gn2"]["value"]).all() and eigs["delta_eig/gn2"]["value"].shape == (len(eig_steps), 2)
    assert "displacement/delta/params/0" in eigs and "displacement/delta/params/1" in eigs  # total norms: a panel per group, a line per eval set
    assert len(results["groups"][0]["runs"]) == 2


def test_newton(tmp_path):
    """The newton optimizer (damped Newton by MINRES: a second-order step with aux state and an implicit backward) runs
    through train (not vmapped: no retrain) and magic."""
    d = str(tmp_path)
    runner = CliRunner()
    for args in [
        ["new", d, "--dataset", "majority", "--model", "mlp", "--optimizer", "newton", "--lr", "1", "--damping", "2", "--n-steps", "6", "--dtype", "fp64"],
        ["train", d, "--num-top-eigs", "1", "--eig-every", "2", "--eval-every", "3", "--n-test", "20", "--no-show"],
        ["magic", d, "--n-test", "5", "--no-show"],
    ]:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    final = torch.load(tmp_path / "full_model" / "final.pt", weights_only=True)
    assert set(final) == {"params", "lr", "wd", "ema_decay", "ema_params", "damping", "rtol", "newton", "eigvec", "stats"}
    assert final["damping"] == 2 and 0 < final["stats"]["newton_iters"] < 100 and torch.isfinite(final["stats"]["residual"])  # converged
    lin = torch.load(tmp_path / "magic" / "5_examples_0" / "linearization.pt", weights_only=True)
    assert torch.isfinite(lin["beta"]).all() and (lin["beta"] != 0).any()
    logged = torch.load(tmp_path / "full_model" / "metrics.pt", weights_only=True)
    assert {"newton_iters", "eig_iters", "residual", "min_eig"} <= set(logged) and len(logged["min_eig"]["step"]) == 6  # the stats
    assert "tangent/params/out.weight" not in logged and torch.isfinite(logged["eig/1"]["value"]).all()
    assert logged["eig/1"]["step"].tolist() == [4] and logged["eig/-1"]["step"].tolist() == list(range(6))  # LOBPCG from start + 2 eig_every; the step's own min eig
    torch.testing.assert_close(logged["eig/-1"]["value"], logged["min_eig"]["value"])  # unscaled
    torch.testing.assert_close(logged["eig/threshold"]["value"], 2 * ((-3 * logged["min_eig"]["value"]).clamp_min(0) + 2))  # 2 / (lr / shift), lr 1, damping 2
    magic_metrics = torch.load(tmp_path / "magic" / "metrics.pt", weights_only=True)
    assert not any(k.startswith("delta/5 examples #0/newton") for k in magic_metrics)  # no norms of the aux groups
