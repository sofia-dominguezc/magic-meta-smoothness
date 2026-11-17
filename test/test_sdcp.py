import pytest
import torch

from src.sdcp import sdcp, symmetric_basis


def problem(k: int, seed: int, shift: float = 0.0):
    """A random symmetric A + shift * I [k, k] and PSD operator B [k, k, k, k] (condition number ~100)"""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(k, k, generator=g)
    basis = symmetric_basis(k)
    root = torch.randn(basis.shape[1], basis.shape[1], generator=g) * torch.logspace(0, -1, basis.shape[1])
    return (A + A.T) / 2 + shift * torch.eye(k), (basis @ root @ root.T @ basis.T).reshape(k, k, k, k)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_sdcp_is_complementary(k, float64):
    """X and A + B[X] are PSD and orthogonal, whatever the rank of X (0 when A is PSD); the closed form for k = 1."""
    ranks = set()
    for seed in range(12):
        A, B = problem(k, seed, shift=[-2.0, 0.0, 1.0, 5.0][seed % 4])
        X = sdcp(A, B)
        slack = A + torch.einsum("ijkl,kl->ij", B, X)
        scale = A.norm() * (X.norm() + A.norm() / B.norm())
        assert torch.linalg.eigvalsh(X).min() >= -1e-10 * X.norm() and torch.linalg.eigvalsh(slack).min() >= -1e-10 * A.norm()
        assert (X * slack).sum().abs() <= 1e-10 * scale
        ranks.add(int((torch.linalg.eigvalsh(X) > 1e-8 * X.norm().clamp_min(1e-30)).sum()))
        if seed % 4 == 3:
            assert torch.linalg.eigvalsh(A).min() > 0 and (X == 0).all()
        if k == 1:
            torch.testing.assert_close(X, (-A / B[0, 0]).clamp_min(0))
    assert 0 in ranks and max(ranks) > 0 and (k == 1 or any(0 < rank < k for rank in ranks))  # incl. rank-deficient solutions


@pytest.mark.parametrize("k, seed, shift", [(1, 0, -1.0), (2, 1, 0.0), (3, 2, 0.0), (4, 3, 0.3), (4, 4, -3.0)])
def test_sdcp_derivative_matches_finite_differences(k, seed, shift, float64):
    """d<C, X>/d(A, B) by the implicit function theorem against central finite differences."""
    A, B = problem(k, seed, shift)
    A.requires_grad_(), B.requires_grad_()
    g = torch.Generator().manual_seed(seed)
    C, dA = torch.randn(k, k, generator=g), torch.randn(k, k, generator=g)
    basis = symmetric_basis(k)
    dB = torch.randn(basis.shape[1], basis.shape[1], generator=g)
    dA, dB = dA + dA.T, (basis @ (dB + dB.T) @ basis.T).reshape(k, k, k, k)
    grad_A, grad_B = torch.autograd.grad((C * sdcp(A, B)).sum(), (A, B))
    eps = 1e-6
    finite_diff = (C * (sdcp(A + eps * dA, B + eps * dB) - sdcp(A - eps * dA, B - eps * dB))).sum() / (2 * eps)
    torch.testing.assert_close((grad_A * dA).sum() + (grad_B * dB).sum(), finite_diff, rtol=1e-5, atol=0)
