from dataclasses import replace

import pytest
import torch
from torch.func import functional_call

from src.lobpcg import flatten, hessian_eigs, unflatten
from src.minres import minres
from src.utils import cross_entropy, dict_zip, hvp_from_grads
from test.test_magic import setup


@pytest.mark.parametrize("indefinite", [False, True])
def test_minres_matches_dense(indefinite, float64):
    """MINRES against a dense solve on a positive definite and on an indefinite symmetric system (where CG breaks down; the
    solution flips the sign of the negative-curvature components, unlike |A|^-1): fixed iterations, the residual estimate,
    early stopping, warm start, preconditioner."""
    g = torch.Generator().manual_seed(0)
    n, m, shift = 40, 3, 0.5
    eigs = torch.linspace(0.1, 10, n)
    if indefinite:
        eigs[: n // 3] = -eigs[: n // 3] - 1  # the shift leaves eigenvalues of both signs
    assert (eigs + shift).abs().min() > 0.5
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    A = Q @ torch.diag(eigs) @ Q.T
    B = torch.randn(n, m, generator=g)
    B[:, -1] = 0  # a zero column stays zero
    expected = torch.linalg.solve(A + shift * torch.eye(n), B)

    X, res, n_iter = minres(A.__matmul__, B, shift=shift, rtol=0, max_iter=10)  # partial: the residual estimate is exact
    assert n_iter == 10
    torch.testing.assert_close(res, torch.linalg.norm(B - (A @ X + shift * X), dim=0), rtol=1e-6, atol=1e-12)
    X, res, n_iter = minres(A.__matmul__, B, shift=shift, rtol=0, max_iter=n)  # in exact arithmetic n iterations are exact
    torch.testing.assert_close(X, expected, rtol=1e-8, atol=1e-8)

    X, res, n_iter = minres(A.__matmul__, B, shift=shift, rtol=1e-8, max_iter=n)  # error <= residual / min |eig + shift|
    assert n_iter < n and (res <= 1e-8 * torch.linalg.norm(B, dim=0)).all()
    torch.testing.assert_close(X, expected, rtol=1e-5, atol=1e-6)
    x0 = expected + 1e-4 * torch.randn(n, m, generator=g)
    X, _, n_warm = minres(A.__matmul__, B, shift=shift, rtol=1e-8, max_iter=n, x0=x0)
    assert n_warm < n_iter
    torch.testing.assert_close(X, expected, rtol=1e-5, atol=1e-6)
    M = 1 / (A.diagonal() + shift).abs()  # Jacobi (|.| keeps it positive definite on the indefinite system)
    X, res, n_iter = minres(A.__matmul__, B, shift=shift, rtol=1e-8, max_iter=n, precond=lambda V: M[:, None] * V)
    torch.testing.assert_close(X, expected, rtol=1e-5, atol=1e-6)
    x, res, _ = minres(A.__matmul__, B[:, 0], shift=shift, rtol=1e-8, max_iter=n)  # a vector: a vector back
    torch.testing.assert_close(x, expected[:, 0], rtol=1e-5, atol=1e-6)
    assert x.shape == (n,) and res.shape == ()


def test_minres_damped_hessian(float64):
    """(H + shift I)^-1 G on the tiny MLP with `hvp_fn` as the black box (flat <-> dict translation as in `hessian_eigs`)
    against the dense Hessian; the damping leaves H + shift I indefinite."""
    magic, _, _, state = setup("sgd")
    X, Y, _ = magic.loader.get_batch(0)
    params = {k: p.clone().requires_grad_() for k, p in state.params.items()}
    sizes = [p.numel() for p in params.values()]

    def loss(flat: torch.Tensor) -> torch.Tensor:
        unflat = {k: v.view_as(p) for (k, p), v in zip(params.items(), flat.split(sizes))}
        return cross_entropy(functional_call(magic.model, unflat, (X,)), Y).mean()

    flat = torch.cat([p.detach().flatten() for p in params.values()])
    H = torch.func.hessian(loss)(flat)
    eigs = torch.linalg.eigvalsh(H)  # ascending; many are negative at init
    neg = eigs[eigs < 0]
    i = (neg[1:] - neg[:-1]).argmax()
    shift = -(neg[i] + neg[i + 1]) / 2  # the middle of the largest gap between negative eigenvalues: indefinite, nonsingular
    assert (eigs + shift).abs().min() > 1e-2 * eigs[-1]

    hvp_fn = magic.hvp_fn(params, X, Y)  # V {name: [m, *shape]} -> H V

    def A(V: torch.Tensor) -> torch.Tensor:  # [#params, m] -> [#params, m]
        unflat = {k: v.T.reshape(-1, *p.shape) for (k, p), v in zip(params.items(), V.split(sizes))}
        return torch.cat([v.flatten(1) for v in hvp_fn(unflat).values()], dim=1).T

    G = torch.randn(sum(sizes), 2, generator=torch.Generator().manual_seed(1))
    expected = torch.linalg.solve(H + shift * torch.eye(sum(sizes)), G)
    Xs, res, n_iter = minres(A, G, shift=shift, rtol=1e-8, max_iter=sum(sizes))
    assert n_iter < sum(sizes)
    torch.testing.assert_close(Xs, expected, rtol=1e-5, atol=1e-6)


def newton_step_unrolled(s, t: int, grads: dict[str, torch.Tensor]):
    """`NewtonState.step` with the graph kept through the MINRES iterations (HVPs with create_graph) and through the
    shift (the Rayleigh quotient of the detached eigenvector): autograd through it is the exact derivative of the truncated
    solve, which `NewtonState.vjp` (implicit) matches up to the tolerance."""
    hvp_fn, shapes = hvp_from_grads(grads, s.params), {k: p.shape for k, p in s.params.items()}
    _, v, _ = hessian_eigs(hvp_fn, s.eigvec, sign=-1, batched=False)
    min_eig = torch.stack([(v_k * hv).sum() for v_k, hv in dict_zip(v, hvp_fn(v, create_graph=True)).values()]).sum()
    shift = (-2 * min_eig).clamp_min(0) + s.damping
    rhs = flatten({k: g + s.wd * p for k, (p, g) in dict_zip(s.params, grads).items()}, batched=False)
    A = lambda V: flatten(hvp_fn(unflatten(V, shapes), create_graph=True))
    X, _, _ = minres(A, rhs, shift=shift, rtol=s.rtol, x0=flatten(s.newton, batched=False))
    x = unflatten(X, shapes)
    params = {k: p - s.lr(t) * x_k for k, (p, x_k) in dict_zip(s.params, x).items()}
    return replace(s, params=params, newton={k: x_k.detach() for k, x_k in x.items()}, eigvec=v)  # the same warm starts as `step`


def test_newton_state(float64):
    """The damped Newton step moves by lr (H + (wd + shift) I)^-1 (g + wd p) (dense Hessian) with the shift from the most
    negative eigenvalue, and saves the direction and the stats (aux: no MAGIC state); its `vjp` matches autograd through
    `newton_step_unrolled`, and so does MAGIC through a training of such steps."""
    magic, train_ds, test_loader, state = setup("newton")
    assert state.groups == ("params",) and state.aux == ("newton", "eigvec", "stats") and len(list(state)) == len(state.params)
    X, Y, idx = magic.loader.get_batch(0)
    sizes = [p.numel() for p in state.params.values()]
    s = state.apply(lambda t: t.clone().requires_grad_())
    w = torch.ones(len(train_ds), requires_grad=True)
    loss = (w[idx] * cross_entropy(functional_call(magic.model, s.params, (X,)), Y)).mean()
    grads = torch.autograd.grad(loss, s.params, create_graph=True)

    def flat_loss(flat: torch.Tensor) -> torch.Tensor:
        unflat = {k: v.view_as(p) for (k, p), v in zip(s.params.items(), flat.split(sizes))}
        return cross_entropy(functional_call(magic.model, unflat, (X,)), Y).mean()

    H = torch.func.hessian(flat_loss)(torch.cat([p.detach().flatten() for p in s.params.values()]))
    rhs = torch.cat([(g + 0.01 * p).detach().flatten() for p, g in dict_zip(s.params, grads).values()])
    new = s.step(0, grads)
    min_eig = torch.linalg.eigvalsh(H)[0]
    assert min_eig < 0 and state._shift(new.stats["min_eig"]) > state.damping  # the shift is active: the Rayleigh term of `vjp` is tested
    torch.testing.assert_close(new.stats["min_eig"], min_eig, rtol=0.1, atol=0)  # (LOBPCG rtol, from a random start)
    x = torch.linalg.solve(H + state._shift(new.stats["min_eig"]) * torch.eye(len(rhs)), rhs)
    torch.testing.assert_close(torch.cat([(p - n).flatten() for p, n in dict_zip(s.params, new.params).values()]), x, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(torch.cat([v.flatten() for v in new.newton.values()]), x, rtol=1e-7, atol=1e-9)
    assert 0 < new.stats["newton_iters"] < len(rhs) and new.stats["residual"] <= 1e-10 * rhs.norm()
    assert not new.newton["out.weight"].requires_grad and new.newton["out.weight"].shape == s.params["out.weight"].shape
    assert not s.newton["out.weight"].requires_grad and s.params["out.weight"].requires_grad  # `apply` keeps the aux detached

    g = torch.Generator().manual_seed(0)
    delta = new.apply(lambda t: torch.randn(2, *t.shape, generator=g))  # 2 cotangents on the new state (aux ignored)
    expected = torch.autograd.grad(list(newton_step_unrolled(s, 0, grads)), [*s, w], list(delta), is_grads_batched=True, retain_graph=True)
    cotangents = s.vjp(0, grads, [*s, w], delta)  # (frees the graph: the last use of `grads` in backprop too)
    assert len(cotangents) == len(expected) == len(sizes) + 1
    for a, b in zip(cotangents, expected):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-8)

    n_steps, eval_sets = 3, [torch.arange(10), torch.arange(20, 50)]
    outputs, betas = magic.backprop(state, n_steps=n_steps, eval_sets=[test_loader.dataset.examples(idx) for idx in eval_sets])
    w = torch.ones(len(train_ds), requires_grad=True)
    s = state.apply(lambda t: t.clone().requires_grad_())
    for t in range(n_steps):
        X, Y, idx = magic.loader.get_batch(t)
        loss = (w[idx] * cross_entropy(functional_call(magic.model, s.params, (X,)), Y)).mean()
        s = newton_step_unrolled(s, t, torch.autograd.grad(loss, s.params, create_graph=True))
    for k, idx in enumerate(eval_sets):
        X, Y = test_loader.dataset.examples(idx)
        f = cross_entropy(functional_call(magic.model, s.params, (X,)), Y).mean()
        (expected,) = torch.autograd.grad(f, w, retain_graph=True)
        torch.testing.assert_close(outputs[k], f.detach())
        torch.testing.assert_close(betas[k], expected, rtol=1e-6, atol=1e-8)
