"""MINRES (`minres`): a black box V -> (A + shift I)^-1 V for a symmetric (indefinite ok) operator A given only as V -> A V,
translated into this repo's style from GPyTorch's linear_operator (MIT):
https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/utils/minres.py
(the shift is folded into the operator so that it stays (A + shift I) with a preconditioner; one shift; the residual norm
is returned and checked every iteration; `x0` added). Differentiable through its iterations, for tests."""

from typing import Callable

import torch
from torch import Tensor

Operator = Callable[[Tensor], Tensor]  # [n, m] -> [n, m], evaluated on each column (like `torch_lobpcg`'s A)


def minres(
    A: Operator, B: Tensor, shift: float | Tensor = 0.0, rtol: float = 1e-4, max_iter: int = 100,
    x0: Tensor | None = None, precond: Operator | None = None,
) -> tuple[Tensor, Tensor, int]:
    """Solve (A + shift I) X = B by MINRES (Lanczos + Givens QR of the tridiagonal) for the columns of B [n, m], or a vector [n].
    `precond` = an approximate inverse, SPD. Stops once every column's residual is <= rtol ||b|| (a sync per iteration).
    Returns (X, the residual norms [m] from the recurrence (the M-norm if preconditioned), iterations run).
    """
    eps, tiny = torch.finfo(B.dtype).eps, torch.finfo(B.dtype).tiny
    vector = B.ndim == 1
    if vector:
        B, x0 = B[:, None], None if x0 is None else x0[:, None]
    b_norm = torch.linalg.norm(B, dim=0, keepdim=True)
    b_norm = torch.where(b_norm == 0, 1.0, b_norm)
    B = B / b_norm
    if x0 is None:
        X, R = torch.zeros_like(B), B
    else:
        X = x0 / b_norm
        R = B - (A(X) + shift * X)
    # Lanczos vectors of M^(1/2) (A + shift I) M^(1/2): z plain, q = M z, normalized to z . q = 1 (z = q without M)
    z_prev, z = torch.zeros_like(B), R
    q = z if precond is None else precond(z)
    beta = (z * q).sum(0, keepdim=True).sqrt().clamp_min(eps)  # [1, m]; = ||r|| (the M-norm) of the initial X
    z, q = z / beta, q / beta
    phi = beta  # ||b|| e_1 rotated along with the tridiagonal: the residual norm of the current X
    cos_prev2, sin_prev2 = torch.ones_like(beta), torch.zeros_like(beta)  # Givens rotations from two and one steps back
    cos_prev, sin_prev = torch.ones_like(beta), torch.zeros_like(beta)
    D_prev2, D_prev = torch.zeros_like(B), torch.zeros_like(B)  # search directions = Lanczos vectors times R^-1
    n = 0
    for n in range(1, max_iter + 1):
        Aq = A(q) + shift * q
        alpha = (Aq * q).sum(0, keepdim=True)
        z_next = Aq - alpha * z - beta * z_prev
        q_next = z_next if precond is None else precond(z_next)
        beta_next = (z_next * q_next).sum(0, keepdim=True).sqrt()
        # beta = 0 (up to round-off) is an invariant subspace: the column has converged, its vectors become 0 and stay so
        done = beta_next <= eps * (alpha.abs() + beta)
        beta_next = torch.where(done, 0.0, beta_next)
        safe = torch.where(done, 1.0, beta_next)  # (a division by 0 would make the backward nan, when differentiated through)
        z_next, q_next = torch.where(done, 0.0, z_next / safe), torch.where(done, 0.0, q_next / safe)
        # QR of the tridiagonal: the two previous rotations on its new column (beta, alpha, beta_next), then a new one
        epsilon = sin_prev2 * beta
        delta = cos_prev2 * beta
        gamma = cos_prev * alpha - sin_prev * delta
        delta = sin_prev * alpha + cos_prev * delta
        radius = torch.sqrt(gamma**2 + beta_next**2).clamp_min(tiny)  # 0 only after convergence: then cos = sin = 0
        cos, sin = gamma / radius, beta_next / radius
        D = (q - delta * D_prev - epsilon * D_prev2) / radius
        X = X + (cos * phi) * D
        phi = -sin * phi
        z_prev, z, q, beta = z, z_next, q_next, beta_next
        cos_prev2, sin_prev2, cos_prev, sin_prev = cos_prev, sin_prev, cos, sin
        D_prev2, D_prev = D_prev, D
        if bool((phi.abs() <= rtol).all()):
            break
    X, res = X * b_norm, (phi.abs() * b_norm).squeeze(0)
    return (X[:, 0], res[0], n) if vector else (X, res, n)
