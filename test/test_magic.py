from dataclasses import asdict, replace

import pytest
import torch
from torch.func import functional_call

from src.data import Dataset, Loader, Majority
from src.lobpcg import hessian_eigs
from src.magic import Magic
from src.utils import cross_entropy, hvp_from_grads
from src.models import MLP, Linear, ResNet, ViT
from src.optimizers import AdamState, CentralFlowState, EdgeFlowState, HeunState, LRSchedule, NewtonState, SGDMomentumState, SGDState, SharpnessState
from src.logging import Logger, op_norm
from main import train_logged


def setup(optimizer: str, **kwargs):
    """Tiny majority task (float64 if the `float64` fixture is used) and an initial state."""
    train_ds, test_ds = Majority(True, size=64, dim=8), Majority(False, size=64, dim=8)
    torch.manual_seed(0)
    model = MLP((8,), 2, width=16, depth=1).to(torch.get_default_dtype())
    params = {k: p.detach().clone() for k, p in model.named_parameters()}
    if optimizer == "sgd":
        state = SGDState(params=params, lr=LRSchedule(0.1), wd=0.01)
    elif optimizer == "momentum":
        state = SGDMomentumState(params=params, lr=LRSchedule(0.1), wd=0.01, momentum=0.9)
    elif optimizer == "heun":
        state = HeunState(params=params, lr=LRSchedule(0.1), wd=0.01, alpha=0.75)
    elif optimizer == "sharpness":
        state = SharpnessState(params=params, lr=LRSchedule(1.0), wd=0.01)
    elif optimizer == "edgeflow":
        state = EdgeFlowState(params=params, lr=LRSchedule(0.5), wd=0.01)
    elif optimizer == "centralflow":
        state = CentralFlowState(params=params, lr=LRSchedule(1.0), wd=0.01)
    elif optimizer == "newton":  # rtol: the implicit backward is compared with the unrolled one (equal up to it)
        state = NewtonState(params=params, lr=LRSchedule(1.0), wd=0.01, damping=1.0, rtol=1e-10)
    else:
        state = AdamState(params=params, lr=LRSchedule(0.01), wd=0.01)
    magic = Magic(model=model, loader=Loader(train_ds, batch_size=8, seed=3), loss_fn=cross_entropy, **kwargs)
    return magic, train_ds, Loader(test_ds, batch_size=7), state  # eval sets of several batches


def backprop(magic, state, n_steps, eval_sets, **kwargs):
    """(f(s_T), df/dw) of the training of `n_steps` steps from `state`"""
    outputs, delta = magic.eval_grads(magic.train(state, n_steps=n_steps, pbar=False), eval_sets)
    return outputs, magic.backprop(state, delta, n_steps=n_steps, eval_sets=eval_sets, **kwargs)[1]


def weighted_loss(magic, X, Y, w):
    """The `forward_fn` of a batch with the example weights `w`"""
    return lambda p: ((w * cross_entropy(functional_call(magic.model, p, (X,)), Y)).mean(), None)


@pytest.mark.parametrize("passes", [1, 2, 3])
@pytest.mark.parametrize("n_steps", [1, 4, 11, 27])  # n_steps^(1/passes) exact, and not (uneven last chunk)
def test_replay_reconstructs_training(n_steps, passes):
    magic, _, _, state = setup("momentum")
    forward = [s.apply(lambda t: t.clone()) for _, s, _ in magic.train.trace(state, n_steps=n_steps)][:-1]  # (not the final state)
    replayed = list(magic.replay(state, n_steps=n_steps, passes=passes))
    assert len(replayed) == n_steps
    for a, b in zip(reversed(forward), replayed):
        for x, y in zip(a, b):
            assert torch.equal(x, y)


@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "adam", "heun", "sharpness", "edgeflow"])
def test_magic_matches_autograd_through_training(optimizer, float64):
    """MAGIC's df/dw must equal autograd through the whole unrolled training (for the sharpness step: through the
    Rayleigh quotient of the detached eigenvector, the same expression)."""
    magic, train_ds, test_loader, state = setup(optimizer)
    n_steps, eval_sets = 11, [torch.arange(10), torch.arange(20, 50)]
    outputs, betas = backprop(magic, state, n_steps, [test_loader.dataset.examples(idx) for idx in eval_sets])

    w = torch.ones(len(train_ds), requires_grad=True)
    s = state.apply(lambda t: t.clone().requires_grad_())
    for t in range(n_steps):
        X, Y, idx = magic.loader.get_batch(t)
        s = s.step(t, weighted_loss(magic, X, Y, w[idx]), create_graph=True)[0]
    for k, idx in enumerate(eval_sets):
        X, Y = test_loader.dataset.examples(idx)
        f = cross_entropy(functional_call(magic.model, s.params, (X,)), Y).mean()
        (expected,) = torch.autograd.grad(f, w, retain_graph=True)
        torch.testing.assert_close(outputs[k], f.detach())
        torch.testing.assert_close(betas[k], expected)


def test_centralflow_magic_matches_finite_differences(float64, monkeypatch):
    """MAGIC through central flow steps on the edge of stability (the SDCP solution and the critical eigenvectors
    differentiated implicitly) against central finite differences of the training in the example weights."""
    monkeypatch.setattr(CentralFlowState, "EIG_RTOL", 1e-11)
    monkeypatch.setattr(CentralFlowState, "VJP_RTOL", 1e-11)
    magic, train_ds, test_loader, state = setup("centralflow")
    X, Y, _ = magic.loader.get_batch(0)
    sharpness = hessian_eigs(magic.hvp_fn(state.params, X, Y), state.eigvecs, rtol=1e-8)[0][0]
    state = replace(state, lr=LRSchedule(2.1 / sharpness.item()))  # starts just above the edge 2 / lr
    n_steps, eval_sets = 8, [test_loader.dataset.examples(idx) for idx in (torch.arange(10), torch.arange(20, 50))]
    stats = [s.stats for _, s, _ in magic.train.trace(state, n_steps=n_steps, pbar=False)][1:]
    assert {int(s["n_critical"]) for s in stats} == {1, 2}
    assert any(s["oscillation"] > 0 for s in stats) and any(s["oscillation"] == 0 for s in stats)  # on and below the edge
    _, betas = backprop(magic, state, n_steps, eval_sets)

    direction, eps = torch.randn(len(train_ds), generator=torch.Generator().manual_seed(0)), 1e-6
    finals = [magic.train(state, n_steps=n_steps, weights=1 + sign * eps * direction, pbar=False) for sign in (1, -1)]
    finite_diff = torch.stack([magic.forward(finals[0].params, *e)[0] - magic.forward(finals[1].params, *e)[0] for e in eval_sets]) / (2 * eps)
    torch.testing.assert_close(betas @ direction, finite_diff, rtol=1e-6, atol=0)


@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "adam"])
def test_train_batched_matches_single(optimizer, float64):
    """Training R weightings at once (vmap) gives the same states as training them one by one."""
    magic, train_ds, _, state = setup(optimizer)
    weights = 1 + 0.5 * torch.randn(3, len(train_ds), generator=torch.Generator().manual_seed(0))
    def train(weights):  # -> (final, checkpoints every 2 steps)
        states = [s.apply(lambda t: t.clone()) for _, s, _ in magic.train.trace(state, n_steps=5, weights=weights)]
        return states.pop(), states[::2]

    batched, ckpts = train(weights)
    assert len(ckpts) == 3 and all(t.shape[0] == 3 for t in batched)
    for r in range(3):
        single, single_ckpts = train(weights[r])
        for a, b in zip([single, *single_ckpts], [batched, *ckpts]):
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y[r])


@pytest.mark.parametrize("optimizer", ["momentum", "adam"])
def test_lr_schedule(optimizer, float64):
    """A warmup + cosine lr schedule (in the state, so saved with it) is followed by train (so replay) and backprop
    alike: MAGIC matches autograd through the scheduled training."""
    magic, train_ds, test_loader, state = setup(optimizer)
    n_steps, lr = 5, state.lr.lr
    state = replace(state, lr=LRSchedule(lr, kind="cosine", warmup=2, n_steps=n_steps))
    lrs = [state.lr(t) for t in range(n_steps + 1)]
    assert lrs[0] == lr / 2 and lrs[1] == lrs[2] == lr and lr > lrs[3] > lrs[4] > lrs[5] == pytest.approx(LRSchedule.FINAL_FRACTION * lr)
    assert type(state)(**asdict(state)).lr == state.lr  # survives a save
    final = magic.train(state, n_steps=n_steps)
    outputs, betas = backprop(magic, state, n_steps, [test_loader.dataset.examples(torch.arange(10))])

    w = torch.ones(len(train_ds), requires_grad=True)
    s = state.apply(lambda t: t.clone().requires_grad_())
    for t in range(n_steps):
        X, Y, idx = magic.loader.get_batch(t)
        s = s.step(t, weighted_loss(magic, X, Y, w[idx]), create_graph=True)[0]
    for a, b in zip(final, s):
        torch.testing.assert_close(a, b.detach())
    X, Y = test_loader.dataset.examples(torch.arange(10))
    f = cross_entropy(functional_call(magic.model, s.params, (X,)), Y).mean()
    torch.testing.assert_close(betas[0], torch.autograd.grad(f, w)[0])


def test_train_yields_grads_and_graph(float64):
    """`train.trace` yields (t, s_t, g_t) with g_t the gradient of batch t at s_t (None at the end); with `create_graph`,
    HVPs from it match `hvp_fn` on the batch and backward AD through the step gives (I - lr (H + wd))^T v = the SGD tangent."""
    magic, _, _, state = setup("sgd")
    for t, s, grads in magic.train.trace(state, n_steps=2, create_graph=True):
        if grads is None:
            assert t == 2
            continue
        X, Y, _ = magic.loader.get_batch(t)
        assert all(p.requires_grad and p.is_leaf for p in s.params.values())
        expected = torch.autograd.grad(cross_entropy(functional_call(magic.model, s.params, (X,)), Y).mean(), s.params)
        torch.testing.assert_close(grads, expected)
        V = {k: torch.randn(2, *p.shape) for k, p in s.params.items()}
        torch.testing.assert_close(hvp_from_grads(grads, s.params)(V), magic.hvp_fn(s.params, X, Y)(V))
        v = {k: V_k[0] for k, V_k in V.items()}
        new = s._update(t, grads)
        jtv = torch.autograd.grad(list(new), list(s), list(v.values()), retain_graph=True)
        Hv = hvp_from_grads(grads, s.params)({k: V_k[:1] for k, V_k in V.items()})
        lr, wd = s.lr(t), s.wd
        for a, (k, v_k) in zip(jtv, v.items()):
            torch.testing.assert_close(a, v_k - lr * (Hv[k][0] + wd * v_k))
    for _, s, _ in magic.train.trace(state, n_steps=2):
        assert not any(p.requires_grad for p in s.params.values())


@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "adam"])  # (not the second-order steps: duals through LOBPCG / MINRES are slow)
def test_tangent_matches_finite_differences(float64, optimizer):
    """The tangent logged by train_logged (forward-mode AD through the loss gradient and the step, for any optimizer) has the norms of
    (s_t(s_0 + eps v) - s_t(s_0 - eps v)) / 2 eps, for all the state groups; with the eigenvalues and curvatures logged too."""
    magic, _, test_loader, state = setup(optimizer)
    if optimizer == "adam":  # a positive second moment: the finite differences perturb it in both directions (sqrt)
        state = replace(state, second_moment={k: torch.full_like(p, 1e-2) for k, p in state.params.items()})
    n_steps, eps = 6, 1e-6  # eigenvalues at steps 2 and 4
    logger = Logger(name="test", flush_every=10**9)
    train_logged(magic, state, n_steps=n_steps, test_loader=test_loader, logger=logger, tangent=True, n_top_eigs=2, eig_every=2,
                 curvature=True, eval_every=0)
    g = torch.Generator().manual_seed(0)  # the same v as train_logged
    v = state.apply(lambda s: torch.randn(s.shape, generator=g, dtype=torch.float32).to(s.dtype))
    v = v.apply(lambda s: s / op_norm(s, n_iter=10)[0])

    def trajectory(sign):
        it = iter(v)
        return [s for _, s, _ in magic.train.trace(state.unflatten([s + sign * eps * next(it) for s in state]), n_steps=n_steps, pbar=False)]

    ref = Logger(name="ref", flush_every=10**9)
    for t, (a, b) in enumerate(zip(trajectory(1), trajectory(-1))):
        ref.log_norms(t, "tangent", a.unflatten([(x - y) / (2 * eps) for x, y in zip(a, b)]))
    logger.flush(), ref.flush()
    keys = [k for k in logger._history if k.startswith("tangent/")]
    assert len(keys) == len([k for k in ref._history if k.startswith("tangent/")]) == len(list(state)) and all(len(logger._history[k][0][0]) == n_steps + 1 for k in keys)
    for k in keys:
        torch.testing.assert_close(logger._history[k][0][1], ref._history[k][0][1], rtol=1e-5, atol=1e-8)
    for k in ("eig/1", "eig/2", "grad_hess/Hv", "grad_hess/vHv", "tangent_hess/Hv", "tangent_hess/vHv"):
        assert k in logger._history and torch.isfinite(logger._history[k][0][1]).all(), k


def test_magic_does_not_modify_init_state():
    """On CPU `.to()` doesn't copy, so the replayed s_0 shares tensors with `init_state`."""
    magic, _, test_loader, state = setup("momentum")
    backprop(magic, state, 4, [test_loader.dataset.examples(torch.arange(10))])
    backprop(magic, state, 1, [test_loader.dataset.examples(torch.arange(10))])  # (replays `init_state` itself)
    assert not any(t.requires_grad for t in state)


@pytest.mark.parametrize("passes", [1, 2])
def test_backprop_in_segments(passes, float64):
    """Backpropagating from checkpoint to checkpoint (each segment from the delta the next one returns) = in one go."""
    magic, _, test_loader, state = setup("momentum")
    eval_sets = [test_loader.dataset.examples(idx) for idx in (torch.arange(10), torch.arange(20, 50))]
    bounds = [0, 3, 4, 9]
    states = {0: state}
    for a, b in zip(bounds, bounds[1:]):
        states[b] = magic.train(states[a], n_steps=b - a, start=a, pbar=False)
    outputs, delta = magic.eval_grads(states[9], eval_sets)
    whole_delta, whole = magic.backprop(state, delta, n_steps=9)
    betas = torch.zeros_like(whole)
    for a, b in reversed(list(zip(bounds, bounds[1:]))):
        delta, segment_betas = magic.backprop(states[a], delta, n_steps=b - a, start=a, passes=passes)
        betas += segment_betas
    torch.testing.assert_close(betas, whole)
    for a, b in zip(delta, whole_delta):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(outputs, backprop(magic, state, 9, eval_sets)[0])


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("precond", [False, True])
def test_top_hessian_eigs_match_dense(precond, negative, float64):
    """LOBPCG's top / most negative eigenvalues (HVPs = `hvp_fn`, double backward) against the dense Hessian, plain and preconditioned."""
    magic, _, _, state = setup("sgd")
    X, Y, _ = magic.loader.get_batch(0)
    params = {k: p.clone().requires_grad_() for k, p in state.params.items()}
    sizes = [p.numel() for p in params.values()]
    g = torch.Generator().manual_seed(1)
    P = {k: torch.rand(p.shape, generator=g, dtype=p.dtype) + 0.5 for k, p in params.items()} if precond else None

    def loss(flat: torch.Tensor) -> torch.Tensor:
        unflat = {k: v.view_as(p) for (k, p), v in zip(params.items(), flat.split(sizes))}
        return cross_entropy(functional_call(magic.model, unflat, (X,)), Y).mean()

    flat = torch.cat([p.detach().flatten() for p in params.values()])
    H = torch.func.hessian(loss)(flat)
    if P is not None:
        sqrt_p = torch.cat([p.flatten() for p in P.values()]).sqrt()
        H = sqrt_p[:, None] * H * sqrt_p[None, :]
    all_eigs = torch.linalg.eigvalsh(H)  # ascending
    expected = all_eigs[:3] if negative else all_eigs.flip(0)[:3]
    assert (expected < 0).all() if negative else (expected > 0).all()  # the Hessian is indefinite: both signs tested

    hvp_fn = magic.hvp_fn(params, X, Y)  # V -> H V on batched params dicts
    start = {k: torch.randn(3, *p.shape, generator=g, dtype=p.dtype) for k, p in params.items()}
    eigvals, eigvecs, n_iter = hessian_eigs(hvp_fn, start, P, sign=-1 if negative else 1)
    eigvals, _, _ = hessian_eigs(hvp_fn, eigvecs, P, sign=-1 if negative else 1)  # warm-started, like the logging
    assert 0 < n_iter and sum(sizes) >= 12  # LOBPCG ran (no dense fallback)
    torch.testing.assert_close(eigvals, expected, rtol=5e-2, atol=0)


def test_datasets_independent_of_default_dtype():
    """Batches are in the default dtype at call time and the same (up to rounding) whatever the dtype at construction."""
    torch.set_default_dtype(torch.float64)
    ds = Majority(True, size=16, dim=8)
    X64, Y64 = ds.examples(torch.arange(16))
    torch.set_default_dtype(torch.float32)
    X32, Y32 = ds.examples(torch.arange(16))
    assert X64.dtype == torch.float64 and X32.dtype == torch.float32
    torch.testing.assert_close(X32.double(), X64, rtol=1e-6, atol=0)
    assert torch.equal(Y32, Y64) and torch.equal(Majority(True, size=16, dim=8).labels, ds.labels)


def test_loader_epochs():
    """Within an epoch the batches partition a permutation (no example twice), epochs differ, and a batch depends only
    on its step (not on the order of the calls or the Loader instance)."""
    ds = Majority(True, size=70, dim=4)
    loader, other = Loader(ds, batch_size=16, seed=5), Loader(ds, batch_size=16, seed=5)
    n_batches = 70 // 16
    idx = {t: loader.get_batch(t)[2] for t in [7, 3, 0, 1, 2, 4, 5, 6, 8]}  # 2+ epochs, out of order
    epoch0 = torch.cat([idx[t] for t in range(n_batches)])
    assert len(epoch0) == n_batches * 16 and len(set(epoch0.tolist())) == len(epoch0)
    assert not torch.equal(epoch0, torch.cat([idx[t] for t in range(n_batches, 2 * n_batches)]))
    assert all(torch.equal(idx[t], other.get_batch(t)[2]) for t in idx)
    assert not torch.equal(idx[0], Loader(ds, batch_size=16, seed=6).get_batch(0)[2])


def test_dataset_take():
    """`take` keeps a random subset as examples 0..n-1 that match the originals through `ids`."""
    full, ds = Majority(True, size=64, dim=8), Majority(True, size=64, dim=8)
    ds.take(16, seed=1)
    assert len(ds) == 16 and len(set(ds.ids.tolist())) == 16 and torch.equal(ds.ids, ds.ids.sort().values)
    X, Y = ds.examples(torch.arange(16))
    X_full, Y_full = full.examples(ds.ids)
    assert torch.equal(X, X_full) and torch.equal(Y, Y_full) and torch.equal(Y, ds.labels)


def test_momentum_matches_torch_optim():
    params = {"a": torch.randn(5), "b": torch.randn(3, 2)}
    state = SGDMomentumState(params=params, lr=LRSchedule(0.1), wd=0.01, momentum=0.9)
    torch_params = [p.clone().requires_grad_() for p in params.values()]
    opt = torch.optim.SGD(torch_params, lr=0.1, weight_decay=0.01, momentum=0.9)
    for t in range(3):
        grads = {k: torch.randn_like(p) for k, p in params.items()}
        state = state._update(t, grads)
        for p, g in zip(torch_params, grads.values()):
            p.grad = g
        opt.step()
    for ours, theirs in zip(state.params.values(), torch_params):
        torch.testing.assert_close(ours, theirs.detach())


def test_heun_matches_small_sgd_steps(float64):
    """Both integrate the gradient flow: heun to O(lr^2), far closer than SGD at the same lr."""
    torch.manual_seed(0)
    inputs, targets = torch.randn(20, 5), torch.randn(20)
    forward_fn = lambda params: (((torch.tanh(inputs @ params["a"]) - targets) ** 2).mean(), None)

    def train(state_cls, lr: float, n_steps: int) -> torch.Tensor:
        state = state_cls(params={"a": torch.ones(5, requires_grad=True)}, lr=LRSchedule(lr), wd=0.01)
        for t in range(n_steps):
            state = state.step(t, forward_fn)[0]
        return state.params["a"]

    flow, heun, sgd = train(SGDState, 1e-4, 20_000), train(HeunState, 0.1, 20), train(SGDState, 0.1, 20)
    torch.testing.assert_close(heun, flow, rtol=0, atol=1e-3)
    assert (heun - flow).norm() < 0.1 * (sgd - flow).norm()


def test_adam_matches_torch_optim():
    params = {"a": torch.randn(5), "b": torch.randn(3, 2)}
    state = AdamState(params=params, lr=LRSchedule(0.1), wd=0.01, betas=(0.8, 0.99))
    torch_params = [p.clone().requires_grad_() for p in params.values()]
    groups = [{"params": [p], "lr": 0.1 * p.numel() ** -0.5} for p in torch_params]  # the muP lrs
    opt = torch.optim.Adam(groups, weight_decay=0.01, betas=(0.8, 0.99))
    for t in range(3):
        grads = {k: torch.randn_like(p) for k, p in params.items()}
        state = state._update(t, grads)
        for p, g in zip(torch_params, grads.values()):
            p.grad = g
        opt.step()
    for ours, theirs in zip(state.params.values(), torch_params):
        torch.testing.assert_close(ours, theirs.detach())
    assert all(p.shape == v.shape for p, v in zip(state.params.values(), state.precond(2).values()))


class Noise(Dataset):
    """Random images with random labels, for testing the models."""

    def __init__(self, shape: tuple[int, ...], n: int = 16, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.X, self.labels = torch.randn(n, *shape, generator=g), torch.randint(3, (n,), generator=g)
        self.in_shape, self.out_dim = shape, 3

    def examples(self, idx):
        return self.X[idx], self.labels[idx]


@pytest.mark.parametrize("model_cls, kwargs", [(ResNet, dict(width=4)), (ViT, dict(width=8, head_dim=4, depth=1))])
def test_models_support_logging_and_magic(model_cls, kwargs, tmp_path):
    """Double backward (through the loss and the step) and LOBPCG (needed for the logged metrics and MAGIC) work through the models."""
    train_ds, test_ds = Noise((3, 8, 8)), Noise((3, 8, 8), seed=1)
    torch.manual_seed(0)
    model = model_cls(train_ds.in_shape, train_ds.out_dim, **kwargs)
    state = SGDState(params={k: p.detach().clone() for k, p in model.named_parameters()}, lr=LRSchedule(0.1))
    logger = Logger(name="test")
    magic, test_loader = Magic(model=model, loader=Loader(train_ds, batch_size=4), loss_fn=cross_entropy), Loader(test_ds)
    train_logged(magic, state, n_steps=5, test_loader=test_loader, logger=logger, eval_every=2, n_top_eigs=2, eig_every=2, tangent=True)  # eigs at step 4
    eval_sets = [test_loader.dataset.examples(torch.arange(8))]
    outputs, betas = backprop(magic, state, 3, eval_sets, log=logger)
    delta = magic.eval_grads(magic.train(state, n_steps=3), eval_sets)[1]
    for t, _, delta, d_weights in magic.backprop.trace(state, delta, n_steps=3):
        assert d_weights.shape == (1, 4)
    logger.finish(tmp_path / "metrics.pt")
    metrics = torch.load(tmp_path / "metrics.pt", weights_only=True)
    assert all(torch.isfinite(m["value"]).all() for m in metrics.values())
    assert {"loss", "acc", "eig/1", "eig/2", "test_loss", "test_acc", "params/out.weight",
            "grads/out.bias", "tangent/params/out.weight", "delta/0/params/out.weight", "d_weights/0"} <= set(metrics)
    assert betas.shape == (1, len(train_ds)) and torch.isfinite(betas).all() and (betas != 0).any()
    assert (tmp_path / "metrics.png").exists() and (tmp_path / "metrics_norms.png").exists()


def test_mu_linear_init():
    """The effective weight sqrt(fan_out / fan_in) * w has the LeCun variance 1/fan_in, the raw w an O(1) spectral norm."""
    torch.manual_seed(0)
    for fan_in, fan_out in [(784, 64), (256, 256), (64, 1024)]:
        layer = Linear(fan_in, fan_out)
        torch.testing.assert_close((layer.mult * layer.weight).var().item(), 1 / fan_in, rtol=0.1, atol=0)
        spectral = torch.linalg.matrix_norm(layer.weight, 2)
        assert 0.5 < spectral < 5
        torch.testing.assert_close(op_norm(layer.weight, n_iter=100)[0], spectral, rtol=1e-2, atol=0)
    readout = Linear(256, 10, readout=True)
    torch.testing.assert_close((readout.mult * readout.weight).var().item(), 1 / 256**2, rtol=0.2, atol=0)


@pytest.mark.parametrize("loss", ["ce", "mse"])
def test_gauss_newton_matches_explicit(float64, loss):
    """`Magic.gnvp_fn` is J^T H_L J / B (explicit Jacobians of a tiny MLP), batched over the vectors; for the MSE it has the Hessian's
    form J^T J / B, and equals the Hessian for a linear model."""
    from src.utils import LOSSES, OUTPUT_HESSIANS
    from src.lobpcg import flatten, unflatten
    train_ds = Majority(True, size=64, dim=8)
    torch.manual_seed(0)
    model = MLP((8,), 2, width=16, depth=1)
    params = {k: p.detach().clone() for k, p in model.named_parameters()}
    magic = Magic(model=model, loader=Loader(train_ds, batch_size=16), loss_fn=LOSSES[loss])
    X, Y, _ = magic.loader.get_batch(0)
    V = {k: torch.randn(3, *p.shape) for k, p in params.items()}
    GV = flatten(magic.gnvp_fn(params, X, Y)(V))  # [N, 3]
    # explicit: J [B, C, N] (the params in `flatten`'s order), H_L [B, C, C] as a matrix per example
    out = functional_call(model, params, (X,))
    B, C = out.shape
    jac = torch.func.jacrev(lambda p: functional_call(model, p, (X,)))(params)  # name -> [B, C, *shape]
    J = torch.cat([jac[k].reshape(B, C, -1) for k in params], -1)
    H_L = torch.stack([OUTPUT_HESSIANS[LOSSES[loss]](out, Y, torch.eye(C)[c].expand(B, C)) for c in range(C)], -1)  # [:, :, c] = H_L e_c
    G = torch.einsum("bcn,bcd,bdm->nm", J, H_L, J) / B
    torch.testing.assert_close(GV, G @ flatten(V))
    single = magic.gnvp_fn(params, X, Y)({k: v[0] for k, v in V.items()})
    torch.testing.assert_close(flatten(single, batched=False), GV[:, 0])
    traced = torch.compile(magic.gnvp_fn(params, X, Y), backend="eager", fullgraph=True)  # dynamo traces it as one graph (no C++ compiler here)
    torch.testing.assert_close(flatten(traced(V)), GV)
    if loss == "mse":  # linear model: Gauss-Newton = Hessian
        lin = Linear(8, 2)
        p_lin = {k: p.detach().clone() for k, p in lin.named_parameters()}
        m2 = Magic(model=lin, loader=magic.loader, loss_fn=LOSSES[loss])
        v = {k: torch.randn_like(p) for k, p in p_lin.items()}
        torch.testing.assert_close(m2.gnvp_fn(p_lin, X, Y)(v), dict(m2.hvp_fn(p_lin, X, Y)(v)))


def test_gauss_newton_backprop_matches_exact_for_linear_model(float64):
    """The GN backward (SGD only) against the exact one: for a linear model with the MSE, the Gauss-Newton matrix is the
    Hessian so `gn=True` must match `gn=False`. And against the closed form (I - lr (GN + wd I)) delta for an arbitrary GN product."""
    from src.utils import mse
    train_ds, test_ds = Majority(True, size=64, dim=8), Majority(False, size=64, dim=8)
    torch.manual_seed(0)
    model = Linear(8, 2)
    params = {k: p.detach().clone() for k, p in model.named_parameters()}
    state = SGDState(params=params, lr=LRSchedule(0.1), wd=0.01)
    magic = Magic(model=model, loader=Loader(train_ds, batch_size=8, seed=3), loss_fn=mse)
    eval_sets = [test_ds.examples(torch.arange(10)), test_ds.examples(torch.arange(20, 50))]
    outputs, betas = backprop(magic, state, 7, eval_sets)
    outputs_gn, betas_gn = backprop(magic, state, 7, eval_sets, gn=True)
    final_delta = magic.eval_grads(magic.train(state, n_steps=7), eval_sets)[1]
    torch.testing.assert_close(outputs_gn, outputs)
    torch.testing.assert_close(betas_gn, betas)
    # perturbed steps 3-6 only: the steps t >= 3 are backpropagated as before, the steps t < 3 contribute nothing (df/ds_0 = 0)
    batches = [set(magic.loader.get_batch(t)[2].tolist()) for t in range(7)]
    kept = torch.tensor([not any(i in b for b in batches[:3]) for i in range(len(train_ds))])
    trace = magic.backprop.trace(state, final_delta, n_steps=7, gn=True, perturbed=range(3, 7))
    steps = []
    try:
        while True:
            steps.append(next(trace)[0])
    except StopIteration as e:
        _, betas_late = e.value
    assert steps == [6, 5, 4, 3]
    torch.testing.assert_close(betas_late[:, kept], betas[:, kept])
    assert (betas_late[:, ~kept] == 0).all()
    # perturbed steps 0-1 only: df/ds_t goes through all the steps, df/dw_t is added for t < 2 alone
    _, betas_early = backprop(magic, state, 7, eval_sets, perturbed=range(0, 2))
    early = torch.tensor([any(i in b for b in batches[:2]) for i in range(len(train_ds))])
    only_early = early & torch.tensor([not any(i in b for b in batches[2:]) for i in range(len(train_ds))])
    torch.testing.assert_close(betas_early[:, only_early], betas[:, only_early])
    assert (betas_early[:, ~early] == 0).all() and only_early.any()

    delta = state.apply(lambda t: torch.randn(2, *t.shape))
    gnvp = lambda V, create_graph=False: {k: 3.0 * v for k, v in V.items()}  # any symmetric matrix
    new_delta, d_grads = state.gn_vjp(0, delta, gnvp)
    for k, d in delta.params.items():
        torch.testing.assert_close(new_delta.params[k], d - state.lr(0) * (3.0 * d + state.wd * d))
        torch.testing.assert_close(d_grads[k], -state.lr(0) * d)
    with pytest.raises(AssertionError, match="only implemented for SGD"):
        momentum = SGDMomentumState(params=params, lr=LRSchedule(0.1))
        backprop(magic, momentum, 2, eval_sets, gn=True)
