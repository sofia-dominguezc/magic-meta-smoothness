import pytest
import torch
from torch.func import functional_call

from src.data import Dataset, EvalLoader, Loader, Majority
from src.magic import AdamState, Magic, SGDMomentumState, SGDState, cross_entropy
from src.models import MLP, Linear, ResNet, ViT, adam_lr_mults
from src.utils import Logger, op_norm


def setup(optimizer: str):
    """Tiny majority task (float64 if the `float64` fixture is used) and an initial state."""
    train_ds, test_ds = Majority(True, size=64, dim=8), Majority(False, size=64, dim=8)
    torch.manual_seed(0)
    model = MLP((8,), 2, width=16, depth=1).to(torch.get_default_dtype())
    params = {k: p.detach().clone() for k, p in model.named_parameters()}
    if optimizer == "sgd":
        state = SGDState(params=params, lr=0.1, weight_decay=0.01)
    elif optimizer == "momentum":
        state = SGDMomentumState(params=params, lr=0.1, weight_decay=0.01, momentum=0.9)
    else:
        state = AdamState(params=params, lr=0.01, lr_mults=adam_lr_mults(model), weight_decay=0.01)
    return Magic(model, Loader(train_ds, batch_size=8, seed=3), cross_entropy), train_ds, test_ds, state


@pytest.mark.parametrize("passes", [1, 2, 3])
@pytest.mark.parametrize("n_steps", [1, 4, 11, 27])  # n_steps^(1/passes) exact, and not (uneven last chunk)
def test_rewind_replays_training(n_steps, passes):
    magic, _, _, state = setup("momentum")
    final, forward = magic.forward_pass(state, n_steps=n_steps, ckpt_every=1)
    replayed = list(magic.rewind(state, n_steps=n_steps, passes=passes))
    assert len(replayed) == n_steps + 1
    for a, b in zip([final, *reversed(forward)], replayed):
        for x, y in zip(a, b):
            assert torch.equal(x, y)


@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "adam"])
def test_magic_matches_autograd_through_training(optimizer, float64):
    """MAGIC's df/dw must equal autograd through the whole unrolled training."""
    magic, train_ds, test_ds, state = setup(optimizer)
    n_steps, eval_idxs = 11, {"a": torch.arange(10), "b": torch.arange(20, 50)}
    eval_loaders = {k: EvalLoader(test_ds, idx, batch_size=7) for k, idx in eval_idxs.items()}  # several batches per eval set
    outputs, betas = magic.backward_pass(state, n_steps=n_steps, eval_loaders=eval_loaders)

    w = torch.ones(len(train_ds), requires_grad=True)
    s = state.apply(lambda t: t.clone().requires_grad_())
    for t in range(n_steps):
        X, Y, idx, _ = magic.loader.get(t)
        loss = (w[idx] * cross_entropy(functional_call(magic.model, s.params, (X,)), Y)).mean()
        s = s.step(torch.autograd.grad(loss, s.params, create_graph=True))
    for k, idx in eval_idxs.items():
        X, Y = test_ds.get_batch(idx)
        f = cross_entropy(functional_call(magic.model, s.params, (X,)), Y).mean()
        (expected,) = torch.autograd.grad(f, w, retain_graph=True)
        torch.testing.assert_close(outputs[k], f.detach())
        torch.testing.assert_close(betas[k], expected)


def test_magic_does_not_modify_init_state():
    """On CPU `.to()` doesn't copy, so the replayed s_0 shares tensors with `init_state`."""
    magic, _, test_ds, state = setup("momentum")
    magic.backward_pass(state, n_steps=4, eval_loaders={"a": EvalLoader(test_ds, torch.arange(10))})
    assert not any(t.requires_grad for t in state)


def test_momentum_matches_torch_optim():
    params = {"a": torch.randn(5), "b": torch.randn(3, 2)}
    state = SGDMomentumState(params=params, lr=0.1, weight_decay=0.01, momentum=0.9)
    torch_params = [p.clone().requires_grad_() for p in params.values()]
    opt = torch.optim.SGD(torch_params, lr=0.1, weight_decay=0.01, momentum=0.9)
    for _ in range(3):
        grads = {k: torch.randn_like(p) for k, p in params.items()}
        state = state.step(grads)
        for p, g in zip(torch_params, grads.values()):
            p.grad = g
        opt.step()
    for ours, theirs in zip(state.params.values(), torch_params):
        torch.testing.assert_close(ours, theirs.detach())


def test_adam_matches_torch_optim():
    params = {"a": torch.randn(5), "b": torch.randn(3, 2)}
    state = AdamState(params=params, lr=0.1, weight_decay=0.01, betas=(0.8, 0.99))
    torch_params = [p.clone().requires_grad_() for p in params.values()]
    opt = torch.optim.Adam(torch_params, lr=0.1, weight_decay=0.01, betas=(0.8, 0.99))
    for _ in range(3):
        grads = {k: torch.randn_like(p) for k, p in params.items()}
        state = state.step(grads)
        for p, g in zip(torch_params, grads.values()):
            p.grad = g
        opt.step()
    for ours, theirs in zip(state.params.values(), torch_params):
        torch.testing.assert_close(ours, theirs.detach())
    assert state.t == 3 and all(p.shape == v.shape for p, v in zip(state.params.values(), state.precond.values()))


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
    state = SGDMomentumState(params={k: p.detach().clone() for k, p in model.named_parameters()}, lr=0.1)
    logger = Logger(name="test")
    magic = Magic(model, Loader(train_ds, batch_size=4), cross_entropy, logger, test_loader=EvalLoader(test_ds), eval_every=2, num_top_eigs=2)
    final, _ = magic.forward_pass(state, n_steps=3, log=True)
    outputs, betas = magic.backward_pass(state, n_steps=3, eval_loaders={"f": EvalLoader(test_ds, torch.arange(8))})
    logger.finish(tmp_path / "metrics.pt")
    metrics = torch.load(tmp_path / "metrics.pt", weights_only=True)
    assert all(torch.isfinite(m["value"]).all() for m in metrics.values())
    assert {"loss", "acc", "eig/1", "eig/2", "test loss", "params/out.weight", "grads/out.bias", "tangent/buffer/out.weight", "delta/f/params/out.weight"} <= set(metrics)
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
