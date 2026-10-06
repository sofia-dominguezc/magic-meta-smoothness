import pytest
import torch
from torch.func import functional_call

from src.data import majority
from src.magic import LOSSES, Loader, SGDMomentumState, SGDState, batch_loss, magic, reversed_states, train
from src.models import MLP

loss_fn = LOSSES["mse"]


def setup(optimizer: str):
    """Tiny majority task (float64 if the `float64` fixture is used) and an initial state."""
    train_ds, test_ds = majority(True, "cpu", size=64, dim=8), majority(False, "cpu", size=64, dim=8)
    torch.manual_seed(0)
    model = MLP(8, 1, width=16, depth=1).to(torch.get_default_dtype())
    params = {k: p.detach().clone() for k, p in model.named_parameters()}
    if optimizer == "sgd":
        state = SGDState(params=params, lr=0.1, weight_decay=0.01)
    else:
        state = SGDMomentumState(params=params, lr=0.1, weight_decay=0.01, momentum=0.9)
    return model, Loader(train_ds, batch_size=8, seed=3), train_ds, test_ds, state


@pytest.mark.parametrize("n_steps", [1, 4, 11])  # sqrt(n_steps) exact, and not (uneven last chunk)
def test_reversed_states_replay_training(n_steps):
    model, loader, _, _, state = setup("momentum")
    final, forward = train(model, loader, loss_fn, state, n_steps=n_steps, ckpt_every=1, progress=False)
    replayed = list(reversed_states(model, loader, loss_fn, state, n_steps=n_steps))
    assert len(replayed) == n_steps + 1
    for a, b in zip([final, *reversed(forward)], replayed):
        for x, y in zip(a.all_tensors, b.all_tensors):
            assert torch.equal(x, y)


@pytest.mark.parametrize("optimizer", ["sgd", "momentum"])
def test_magic_matches_autograd_through_training(optimizer, float64):
    """MAGIC's df/dw must equal autograd through the whole unrolled training."""
    model, loader, train_ds, test_ds, state = setup(optimizer)
    n_steps, eval_idxs = 11, [torch.arange(10), torch.arange(20, 50)]
    _, outputs, betas = magic(model, loader, loss_fn, state, n_steps, test_ds, eval_idxs)

    w = torch.ones(len(train_ds), requires_grad=True)
    s = state.apply(lambda t: t.clone().requires_grad_())
    for t in range(n_steps):
        batch, idx, _ = loader.get(t)
        loss, _ = batch_loss(model, loss_fn, s.params, batch, w[idx])
        s = s.step(torch.autograd.grad(loss, s.params, create_graph=True))
    for idx, output, beta in zip(eval_idxs, outputs, betas):
        test = test_ds[idx]
        f = loss_fn(functional_call(model, s.params, (test["x"],)), test).mean()
        (expected,) = torch.autograd.grad(f, w, retain_graph=True)
        torch.testing.assert_close(output, f.detach())
        torch.testing.assert_close(beta, expected)


def test_magic_does_not_modify_init_state():
    """On CPU `.to()` doesn't copy, so the replayed s_0 shares tensors with `init_state`."""
    model, loader, _, test_ds, state = setup("momentum")
    magic(model, loader, loss_fn, state, 4, test_ds, [torch.arange(10)])
    assert not any(t.requires_grad for t in state.all_tensors)


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
