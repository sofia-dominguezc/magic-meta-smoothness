from dataclasses import asdict, replace

import pytest
import torch
from torch.func import functional_call

from src.data import Dataset, Loader, Majority
from src.eigvals import hessian_eigs
from src.magic import Magic
from src.utils import cross_entropy, hvp_fn
from src.models import MLP, AdamState, Linear, LRSchedule, ResNet, SGDMomentumState, SGDState, ViT
from src.logging import Logger, op_norm
from train_logged import train_logged


def setup(optimizer: str, **kwargs):
    """Tiny majority task (float64 if the `float64` fixture is used) and an initial state."""
    train_ds, test_ds = Majority(True, size=64, dim=8), Majority(False, size=64, dim=8)
    torch.manual_seed(0)
    model = MLP((8,), 2, width=16, depth=1).to(torch.get_default_dtype())
    params = {k: p.detach().clone() for k, p in model.named_parameters()}
    if optimizer == "sgd":
        state = SGDState(params=params, lr=LRSchedule(0.1), weight_decay=0.01)
    elif optimizer == "momentum":
        state = SGDMomentumState(params=params, lr=LRSchedule(0.1), weight_decay=0.01, momentum=0.9)
    else:
        state = AdamState(params=params, lr=LRSchedule(0.01), weight_decay=0.01)
    magic = Magic(model=model, loader=Loader(train_ds, batch_size=8, seed=3), loss_fn=cross_entropy, **kwargs)
    return magic, train_ds, Loader(test_ds, batch_size=7), state  # eval sets of several batches


@pytest.mark.parametrize("passes", [1, 2, 3])
@pytest.mark.parametrize("n_steps", [1, 4, 11, 27])  # n_steps^(1/passes) exact, and not (uneven last chunk)
def test_replay_reconstructs_training(n_steps, passes):
    magic, _, _, state = setup("momentum")
    forward = [s.apply(lambda t: t.clone()) for _, s, _ in magic.train.trace(state, n_steps=n_steps)]
    final = forward.pop()
    replayed = list(magic.replay(state, n_steps=n_steps, passes=passes))
    assert len(replayed) == n_steps + 1
    for a, b in zip([final, *reversed(forward)], replayed):
        for x, y in zip(a, b):
            assert torch.equal(x, y)


@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "adam"])
def test_magic_matches_autograd_through_training(optimizer, float64):
    """MAGIC's df/dw must equal autograd through the whole unrolled training."""
    magic, train_ds, test_loader, state = setup(optimizer)
    n_steps, eval_sets = 11, {"a": torch.arange(10), "b": torch.arange(20, 50)}
    outputs, betas = magic.backprop(state, n_steps=n_steps, test_loader=test_loader, eval_sets=eval_sets)

    w = torch.ones(len(train_ds), requires_grad=True)
    s = state.apply(lambda t: t.clone().requires_grad_())
    for t in range(n_steps):
        X, Y, idx = magic.loader.get_batch(t)
        loss = (w[idx] * cross_entropy(functional_call(magic.model, s.params, (X,)), Y)).mean()
        s = s.step(t, torch.autograd.grad(loss, s.params, create_graph=True))
    for k, idx in eval_sets.items():
        X, Y = test_loader.dataset.examples(idx)
        f = cross_entropy(functional_call(magic.model, s.params, (X,)), Y).mean()
        (expected,) = torch.autograd.grad(f, w, retain_graph=True)
        torch.testing.assert_close(outputs[k], f.detach())
        torch.testing.assert_close(betas[k], expected)


@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "adam"])
def test_train_batched_matches_single(optimizer, float64):
    """Training R weightings at once (vmap) gives the same states as training them one by one (batches of 1)."""
    magic, train_ds, _, state = setup(optimizer)
    weights = 1 + 0.5 * torch.randn(3, len(train_ds), generator=torch.Generator().manual_seed(0))
    def train(weights):  # -> (final, checkpoints every 2 steps)
        states = [s.apply(lambda t: t.clone()) for _, s, _ in magic.train.trace(state, n_steps=5, weights=weights)]
        return states.pop(), states[::2]

    batched, ckpts = train(weights)
    assert len(ckpts) == 3 and all(t.shape[0] == 3 for t in batched)
    for r in range(3):
        single, single_ckpts = train(weights[r : r + 1])
        for a, b in zip([single, *single_ckpts], [batched, *ckpts]):
            for x, y in zip(a, b):
                torch.testing.assert_close(x[0], y[r])


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
    outputs, betas = magic.backprop(state, n_steps=n_steps, test_loader=test_loader, eval_sets={"a": torch.arange(10)})

    w = torch.ones(len(train_ds), requires_grad=True)
    s = state.apply(lambda t: t.clone().requires_grad_())
    for t in range(n_steps):
        X, Y, idx = magic.loader.get_batch(t)
        loss = (w[idx] * cross_entropy(functional_call(magic.model, s.params, (X,)), Y)).mean()
        s = s.step(t, torch.autograd.grad(loss, s.params, create_graph=True))
    for a, b in zip(final, s):
        torch.testing.assert_close(a, b.detach())
    X, Y = test_loader.dataset.examples(torch.arange(10))
    f = cross_entropy(functional_call(magic.model, s.params, (X,)), Y).mean()
    torch.testing.assert_close(betas["a"], torch.autograd.grad(f, w)[0])


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
        torch.testing.assert_close(hvp_fn(magic.model, cross_entropy, s.params, X, Y, grads=grads)(V), hvp_fn(magic.model, cross_entropy, s.params, X, Y)(V))
        v = {k: V_k[0] for k, V_k in V.items()}
        new = s.step(t, grads)
        jtv = torch.autograd.grad(list(new), list(s), list(v.values()), retain_graph=True)
        Hv = hvp_fn(magic.model, cross_entropy, s.params, X, Y, grads=grads)({k: V_k[:1] for k, V_k in V.items()})
        lr, wd = s.lr(t), s.weight_decay
        for a, (k, v_k) in zip(jtv, v.items()):
            torch.testing.assert_close(a, v_k - lr * (Hv[k][0] + wd * v_k))
    for _, s, _ in magic.train.trace(state, n_steps=2):
        assert not any(p.requires_grad for p in s.params.values())


def test_tangent_matches_finite_differences(float64):
    """The tangent logged by train_logged (backward AD through the SGD step) has the norms of (s_t(s_0 + eps v) - s_t(s_0 - eps v)) / 2 eps."""
    magic, _, test_loader, state = setup("sgd")
    n_steps, eps = 4, 1e-6
    logger = Logger(name="test", flush_every=10**9)
    train_logged(magic, state, n_steps=n_steps, test_loader=test_loader, logger=logger, n_top_eigs=0, eval_every=0, ema_decay=None)
    g = torch.Generator().manual_seed(0)  # the same v as train_logged
    v = state.apply(lambda s: torch.randn(s.shape, generator=g, dtype=s.dtype) / 1)
    v = v.apply(lambda s: s / op_norm(s, n_iter=10)[0])

    def trajectory(sign):
        it = iter(v)
        return [s for _, s, _ in magic.train.trace(state.apply(lambda s: s + sign * eps * next(it)), n_steps=n_steps, pbar=False)]

    ref = Logger(name="ref", flush_every=10**9)
    for t, (a, b) in enumerate(zip(trajectory(1), trajectory(-1))):
        ref.log_norms(t, "tangent", a.unflatten([(x - y) / (2 * eps) for x, y in zip(a, b)]))
    logger.flush(), ref.flush()
    keys = [k for k in logger._history if k.startswith("tangent/")]
    assert len(keys) == len(ref._history) == len(list(state)) and all(len(logger._history[k][0][0]) == n_steps + 1 for k in keys)
    for k in keys:
        torch.testing.assert_close(logger._history[k][0][1], ref._history[k][0][1], rtol=1e-5, atol=1e-8)


def test_magic_does_not_modify_init_state():
    """On CPU `.to()` doesn't copy, so the replayed s_0 shares tensors with `init_state`."""
    magic, _, test_loader, state = setup("momentum")
    magic.backprop(state, n_steps=4, test_loader=test_loader, eval_sets={"a": torch.arange(10)})
    assert not any(t.requires_grad for t in state)


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

    hvp = hvp_fn(magic.model, cross_entropy, params, X, Y)  # V -> H V on batched params dicts
    start = {k: torch.randn(3, *p.shape, generator=g, dtype=p.dtype) for k, p in params.items()}
    eigvals, eigvecs, n_iter = hessian_eigs(hvp, start, P, sign=-1 if negative else 1)
    eigvals, _, _ = hessian_eigs(hvp, eigvecs, P, sign=-1 if negative else 1)  # warm-started, like the logging
    assert 0 < n_iter and sum(sizes) >= 12  # LOBPCG ran (no dense fallback)
    torch.testing.assert_close(eigvals, expected, rtol=1e-2, atol=0)


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
    state = SGDMomentumState(params=params, lr=LRSchedule(0.1), weight_decay=0.01, momentum=0.9)
    torch_params = [p.clone().requires_grad_() for p in params.values()]
    opt = torch.optim.SGD(torch_params, lr=0.1, weight_decay=0.01, momentum=0.9)
    for t in range(3):
        grads = {k: torch.randn_like(p) for k, p in params.items()}
        state = state.step(t, grads)
        for p, g in zip(torch_params, grads.values()):
            p.grad = g
        opt.step()
    for ours, theirs in zip(state.params.values(), torch_params):
        torch.testing.assert_close(ours, theirs.detach())


def test_adam_matches_torch_optim():
    params = {"a": torch.randn(5), "b": torch.randn(3, 2)}
    state = AdamState(params=params, lr=LRSchedule(0.1), weight_decay=0.01, betas=(0.8, 0.99))
    torch_params = [p.clone().requires_grad_() for p in params.values()]
    groups = [{"params": [p], "lr": 0.1 * p.numel() ** -0.5} for p in torch_params]  # the muP lrs
    opt = torch.optim.Adam(groups, weight_decay=0.01, betas=(0.8, 0.99))
    for t in range(3):
        grads = {k: torch.randn_like(p) for k, p in params.items()}
        state = state.step(t, grads)
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


@pytest.mark.parametrize("model_cls, kwargs", [(ResNet, dict(width=4)), (ViT, dict(dim=8, head_dim=4, depth=1))])
def test_models_support_logging_and_magic(model_cls, kwargs, tmp_path):
    """Double backward (through the loss and the step) and LOBPCG (needed for the logged metrics and MAGIC) work through the models."""
    train_ds, test_ds = Noise((3, 8, 8)), Noise((3, 8, 8), seed=1)
    torch.manual_seed(0)
    model = model_cls(train_ds.in_shape, train_ds.out_dim, **kwargs)
    state = SGDState(params={k: p.detach().clone() for k, p in model.named_parameters()}, lr=LRSchedule(0.1))
    logger = Logger(name="test")
    magic, test_loader = Magic(model=model, loader=Loader(train_ds, batch_size=4), loss_fn=cross_entropy), Loader(test_ds)
    train_logged(magic, state, n_steps=3, test_loader=test_loader, logger=logger, eval_every=2, n_top_eigs=2, eig_every=2)
    outputs, betas = magic.backprop(state, n_steps=3, test_loader=test_loader, eval_sets={"f": torch.arange(8)}, log=logger)
    for t, _, delta, d_weights in magic.backprop.trace(state, n_steps=3, test_loader=test_loader, eval_sets={"f": torch.arange(8)}):
        assert d_weights.shape == (1, 4)
    logger.finish(tmp_path / "metrics.pt")
    metrics = torch.load(tmp_path / "metrics.pt", weights_only=True)
    assert all(torch.isfinite(m["value"]).all() for m in metrics.values())
    assert {"loss", "acc", "loss/ema", "test_acc/ema", "eig/1", "eig/2", "tangent_eig/2", "ema_eig/1", "test_loss", "params/out.weight",
            "grads/out.bias", "tangent/params/out.weight", "delta/f/params/out.weight", "d_weights/f"} <= set(metrics)
    assert torch.isfinite(betas["f"]).all() and (betas["f"] != 0).any()
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
