from dataclasses import asdict, replace

import pytest
import torch
from torch.func import functional_call

from src.data import Dataset, EvalLoader, Loader, Majority
from src.eigvals import top_hessian_eigs
from src.magic import Magic, cross_entropy
from src.models import MLP, AdamState, Linear, LRSchedule, ResNet, SGDMomentumState, SGDState, ViT
from src.logging import Logger, op_norm


def setup(optimizer: str, freeze_out: bool = False, **kwargs):
    """Tiny majority task (float64 if the `float64` fixture is used) and an initial state."""
    train_ds, test_ds = Majority(True, size=64, dim=8), Majority(False, size=64, dim=8)
    torch.manual_seed(0)
    model = MLP((8,), 2, width=16, depth=1, freeze_out=freeze_out).to(torch.get_default_dtype())
    params = {k: p.detach().clone() for k, p in model.named_parameters()}
    if optimizer == "sgd":
        state = SGDState(params=params, lr=LRSchedule(0.1), weight_decay=0.01)
    elif optimizer == "momentum":
        state = SGDMomentumState(params=params, lr=LRSchedule(0.1), weight_decay=0.01, momentum=0.9)
    else:
        state = AdamState(params=params, lr=LRSchedule(0.01), weight_decay=0.01)
    return Magic(model=model, loader=Loader(train_ds, batch_size=8, seed=3), loss_fn=cross_entropy, **kwargs), train_ds, test_ds, state


@pytest.mark.parametrize("passes", [1, 2, 3])
@pytest.mark.parametrize("n_steps", [1, 4, 11, 27])  # n_steps^(1/passes) exact, and not (uneven last chunk)
def test_replay_reconstructs_training(n_steps, passes):
    magic, _, _, state = setup("momentum")
    final, forward = magic.train(state, n_steps=n_steps, ckpt_every=1)
    replayed = list(magic.replay(state, n_steps=n_steps, passes=passes))
    assert len(replayed) == n_steps + 1
    for a, b in zip([final, *reversed(forward)], replayed):
        for x, y in zip(a, b):
            assert torch.equal(x, y)


@pytest.mark.parametrize("freeze_out", [False, True])
@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "adam"])
def test_magic_matches_autograd_through_training(optimizer, freeze_out, float64):
    """MAGIC's df/dw must equal autograd through the whole unrolled training (with a frozen readout: through the
    scale-dependent learning rates too)."""
    magic, train_ds, test_ds, state = setup(optimizer, freeze_out)
    n_steps, eval_idxs = 11, {"a": torch.arange(10), "b": torch.arange(20, 50)}
    eval_loaders = {k: EvalLoader(test_ds, idx, batch_size=7) for k, idx in eval_idxs.items()}  # several batches per eval set
    outputs, betas = magic.backprop(state, n_steps=n_steps, eval_loaders=eval_loaders)

    w = torch.ones(len(train_ds), requires_grad=True)
    s = state.apply(lambda t: t.clone().requires_grad_())
    for t in range(n_steps):
        X, Y, idx = magic.loader.get(t)
        loss = (w[idx] * cross_entropy(functional_call(magic.model, s.params, (X,)), Y)).mean()
        s = s.step(torch.autograd.grad(loss, s.params, create_graph=True), t)
    for k, idx in eval_idxs.items():
        X, Y = test_ds.get_batch(idx)
        f = cross_entropy(functional_call(magic.model, s.params, (X,)), Y).mean()
        (expected,) = torch.autograd.grad(f, w, retain_graph=True)
        torch.testing.assert_close(outputs[k], f.detach())
        torch.testing.assert_close(betas[k], expected)


@pytest.mark.parametrize("freeze_out", [False, True])
@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "adam"])
def test_train_batched_matches_single(optimizer, freeze_out, float64):
    """Training R weightings at once (vmap) gives the same states as training them one by one (batches of 1)."""
    magic, train_ds, _, state = setup(optimizer, freeze_out)
    weights = 1 + 0.5 * torch.randn(3, len(train_ds), generator=torch.Generator().manual_seed(0))
    batched, ckpts = magic.train(state, n_steps=5, weights=weights, ckpt_every=2)
    assert len(ckpts) == 3 and all(t.shape[0] == 3 for t in batched)
    for r in range(3):
        single, single_ckpts = magic.train(state, n_steps=5, weights=weights[r : r + 1], ckpt_every=2)
        for a, b in zip([single, *single_ckpts], [batched, *ckpts]):
            for x, y in zip(a, b):
                torch.testing.assert_close(x[0], y[r])


@pytest.mark.parametrize("optimizer", ["momentum", "adam"])
def test_lr_schedule(optimizer, float64):
    """A warmup + cosine lr schedule (in the state, so saved with it) is followed by train (so replay) and backprop
    alike: MAGIC matches autograd through the scheduled training."""
    magic, train_ds, test_ds, state = setup(optimizer)
    n_steps, lr = 5, state.lr.lr
    state = replace(state, lr=LRSchedule(lr, kind="cosine", warmup=2, n_steps=n_steps))
    lrs = [state.lr(t) for t in range(n_steps + 1)]
    assert lrs[0] == lr / 2 and lrs[1] == lrs[2] == lr and lr > lrs[3] > lrs[4] > lrs[5] == pytest.approx(LRSchedule.FINAL_FRACTION * lr)
    assert type(state)(**asdict(state)).lr == state.lr  # survives a save
    final, _ = magic.train(state, n_steps=n_steps)
    outputs, betas = magic.backprop(state, n_steps=n_steps, eval_loaders={"a": EvalLoader(test_ds, torch.arange(10))})

    w = torch.ones(len(train_ds), requires_grad=True)
    s = state.apply(lambda t: t.clone().requires_grad_())
    for t in range(n_steps):
        X, Y, idx = magic.loader.get(t)
        loss = (w[idx] * cross_entropy(functional_call(magic.model, s.params, (X,)), Y)).mean()
        s = s.step(torch.autograd.grad(loss, s.params, create_graph=True), t)
    for a, b in zip(final, s):
        torch.testing.assert_close(a, b.detach())
    X, Y = test_ds.get_batch(torch.arange(10))
    f = cross_entropy(functional_call(magic.model, s.params, (X,)), Y).mean()
    torch.testing.assert_close(betas["a"], torch.autograd.grad(f, w)[0])


def test_magic_does_not_modify_init_state():
    """On CPU `.to()` doesn't copy, so the replayed s_0 shares tensors with `init_state`."""
    magic, _, test_ds, state = setup("momentum")
    magic.backprop(state, n_steps=4, eval_loaders={"a": EvalLoader(test_ds, torch.arange(10))})
    assert not any(t.requires_grad for t in state)


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("precond", [False, True])
def test_top_hessian_eigs_match_dense(precond, negative, float64):
    """LOBPCG's top / most negative eigenvalues (and its batched HVP) against the dense Hessian, plain and preconditioned."""
    magic, _, _, state = setup("sgd")
    X, Y, _ = magic.loader.get(0)
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

    grads = torch.autograd.grad(loss(torch.cat([p.flatten() for p in params.values()])), params, create_graph=True)
    eigvals, eigvecs, n_iter = top_hessian_eigs(grads, params, 3, precond=P, negative=negative)
    eigvals, _, _ = top_hessian_eigs(grads, params, 3, eigvecs, precond=P, negative=negative)  # warm-started, like the logging
    assert 0 < n_iter and sum(sizes) >= 12  # LOBPCG ran (no dense fallback)
    torch.testing.assert_close(eigvals, expected, rtol=1e-2, atol=0)


def test_datasets_independent_of_default_dtype():
    """Batches are in the default dtype at call time and the same (up to rounding) whatever the dtype at construction."""
    torch.set_default_dtype(torch.float64)
    ds = Majority(True, size=16, dim=8)
    X64, Y64 = ds.get_batch(torch.arange(16))
    torch.set_default_dtype(torch.float32)
    X32, Y32 = ds.get_batch(torch.arange(16))
    assert X64.dtype == torch.float64 and X32.dtype == torch.float32
    torch.testing.assert_close(X32.double(), X64, rtol=1e-6, atol=0)
    assert torch.equal(Y32, Y64) and torch.equal(Majority(True, size=16, dim=8).labels, ds.labels)


def test_momentum_matches_torch_optim():
    params = {"a": torch.randn(5), "b": torch.randn(3, 2)}
    state = SGDMomentumState(params=params, lr=LRSchedule(0.1), weight_decay=0.01, momentum=0.9)
    torch_params = [p.clone().requires_grad_() for p in params.values()]
    opt = torch.optim.SGD(torch_params, lr=0.1, weight_decay=0.01, momentum=0.9)
    for t in range(3):
        grads = {k: torch.randn_like(p) for k, p in params.items()}
        state = state.step(grads, t)
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
        state = state.step(grads, t)
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

    def get_batch(self, idx):
        return self.X[idx], self.labels[idx]


@pytest.mark.parametrize("model_cls, kwargs", [(ResNet, dict(width=4)), (ViT, dict(dim=8, head_dim=4, depth=1))])
def test_models_support_logging_and_magic(model_cls, kwargs, tmp_path):
    """Forward-mode AD, double backward and LOBPCG (needed for the logged metrics and MAGIC) work through the models."""
    train_ds, test_ds = Noise((3, 8, 8)), Noise((3, 8, 8), seed=1)
    torch.manual_seed(0)
    model = model_cls(train_ds.in_shape, train_ds.out_dim, **kwargs)
    state = SGDMomentumState(params={k: p.detach().clone() for k, p in model.named_parameters()}, lr=LRSchedule(0.1))
    logger = Logger(name="test")
    magic = Magic(model=model, loader=Loader(train_ds, batch_size=4), loss_fn=cross_entropy, logger=logger, test_loader=EvalLoader(test_ds), eval_every=2, n_top_eigs=2)
    final = magic.train_logged(state, n_steps=3)
    outputs, betas = magic.backprop(state, n_steps=3, eval_loaders={"f": EvalLoader(test_ds, torch.arange(8))})
    logger.finish(tmp_path / "metrics.pt")
    metrics = torch.load(tmp_path / "metrics.pt", weights_only=True)
    assert all(torch.isfinite(m["value"]).all() for m in metrics.values())
    assert {"loss", "acc", "eig/1", "eig/2", "test_loss", "params/out.weight", "grads/out.bias", "tangent/buffer/out.weight", "delta/f/params/out.weight"} <= set(metrics)
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
