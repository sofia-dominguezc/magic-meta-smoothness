"""Semidefinite complementarity problems (`sdcp`): the PSD matrix X with A + B[X] PSD and <X, A + B[X]> = 0, which the
central flow solves at every step (Cohen et al. 2024; their solver is CVXPY:
https://github.com/centralflows/centralflows/blob/main/src/sdcp.py). Here by Newton's method on the small dense problem
(from an interior point method's iterates when needed), differentiable in (A, B) without differentiating through its iterations."""

import math
from typing import Iterator

import torch
from torch import Tensor


def symmetric_basis(k: int) -> Tensor:
    """Orthonormal basis of the symmetric k x k matrices, as columns [k * k, k (k + 1) / 2]"""
    rows, cols = torch.triu_indices(k, k)
    pairs = torch.arange(len(rows))
    basis = torch.zeros(k, k, len(rows), dtype=torch.float64)
    basis[rows, cols, pairs] = basis[cols, rows, pairs] = 1.0
    basis = basis.reshape(k * k, -1)
    return basis / basis.norm(dim=0)


def psd_projection(coords: Tensor, basis: Tensor) -> tuple[Tensor, Tensor]:
    """Projection onto the PSD cone of the symmetric matrix `basis @ coords`, and its Jacobian, both in `basis` coordinates"""
    k = math.isqrt(basis.shape[0])
    eigvals, eigvecs = torch.linalg.eigh((basis @ coords).reshape(k, k))
    tiny = torch.finfo(coords.dtype).tiny
    # divided differences of max(., 0) between the eigenvalues
    slopes = (1 + (eigvals[:, None] + eigvals[None, :]) / (eigvals.abs()[:, None] + eigvals.abs()[None, :]).clamp_min(tiny)) / 2
    rotation = basis.T @ torch.kron(eigvecs, eigvecs)  # eigenbasis [k * k] -> basis coordinates
    return rotation @ eigvals.clamp_min(0).diag().flatten(), (rotation * slopes.flatten()) @ rotation.T


def central_path(A_coords: Tensor, B_coords: Tensor, basis: Tensor, decay: float = 0.01, n_points: int = 8) -> Iterator[Tensor]:
    """Interior points converging to X: the minimizers of <A, X> + <X, B[X]> / 2 - weight * logdet(X), by damped Newton,
    for weights decaying from the objective's scale (until X is numerically singular)"""
    k = math.isqrt(basis.shape[0])
    to_matrix = lambda coords: (basis @ coords).reshape(k, k)
    identity = basis.T @ torch.eye(k, dtype=torch.float64).flatten()

    def objective(X: Tensor, weight: Tensor) -> Tensor:
        cholesky, not_pd = torch.linalg.cholesky_ex(to_matrix(X))
        return A_coords @ X + X @ B_coords @ X / 2 - 2 * weight * cholesky.diagonal().log().sum() if not_pd == 0 else torch.tensor(math.inf)

    # start: the minimizer among the multiples of the identity, for a weight = their objective without the barrier
    slope, curvature = A_coords @ identity, identity @ B_coords @ identity
    weight = slope.clamp_max(0) ** 2 / (2 * curvature) + A_coords.norm() ** 2 / torch.linalg.matrix_norm(B_coords, 2)
    X = (-slope + (slope**2 + 4 * curvature * weight * k).sqrt()) / (2 * curvature) * identity
    for _ in range(n_points):
        for _ in range(50):
            inverse, singular = torch.linalg.inv_ex(to_matrix(X))
            if singular:
                return
            grad = A_coords + B_coords @ X - weight * (basis.T @ inverse.flatten())
            hessian = B_coords + weight * (basis.T @ torch.kron(inverse, inverse) @ basis)
            newton = -torch.linalg.lstsq(hessian, grad[:, None]).solution[:, 0]
            decrement = -(grad @ newton)  # twice the predicted decrease
            if not decrement > 1e-2 * weight:
                break
            damping, value = 1.0, objective(X, weight)
            while damping > 1e-10 and objective(X + damping * newton, weight) > value - 0.25 * damping * decrement:
                damping /= 2
            X = X + damping * newton
        yield X
        weight = decay * weight


def solve_sdcp(A_coords: Tensor, B_coords: Tensor, basis: Tensor, tol: float = 1e-13, max_iter: int = 20) -> tuple[Tensor, Tensor]:
    """Newton on the natural residual X - proj(X - step * (A + B[X])) from 0, then from the points of `central_path` until it
    converges; all in `basis` coordinates (float64).
    Returns (X, dX / d(A + B[X]) at fixed X: the implicit derivative of the solution)."""
    identity = torch.eye(len(A_coords), dtype=torch.float64)
    step = 1 / torch.linalg.matrix_norm(B_coords, 2)

    def natural_residual(X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        projection, projection_jacobian = psd_projection(X - step * (A_coords + B_coords @ X), basis)
        return X - projection, identity - projection_jacobian @ (identity - step * B_coords), projection_jacobian

    def newton(X: Tensor) -> tuple[Tensor, Tensor]:
        """Newton steps while the residual improves over two steps -> (the best X, its relative residual)"""
        best, best_error, stalled = X, torch.tensor(math.inf), 0
        for _ in range(max_iter):
            residual, jacobian, _ = natural_residual(X)
            error = residual.norm() / (step * A_coords.norm() + X.norm())
            best, best_error, stalled = (X, error, 0) if error < best_error else (best, best_error, stalled + 1)
            if best_error <= tol or stalled == 2:
                break
            X = X + torch.linalg.lstsq(jacobian, -residual[:, None]).solution[:, 0]
        return best, best_error

    X, error = newton(torch.zeros_like(A_coords))
    for interior in central_path(A_coords, B_coords, basis) if error > tol else ():
        candidate, candidate_error = newton(interior)
        if candidate_error < error:
            X, error = candidate, candidate_error
        if error <= tol:
            break
    _, jacobian, projection_jacobian = natural_residual(X)
    return X, -step * torch.linalg.lstsq(jacobian, projection_jacobian).solution


def sdcp(A: Tensor, B: Tensor) -> Tensor:
    """The PSD X [k, k] with A + B[X] PSD and <X, A + B[X]> = 0, i.e. the minimizer of <A, X> + <X, B[X]> / 2 over the PSD cone.
    A [k, k] symmetric; B [k, k, k, k] a PSD operator on symmetric matrices, B[X]_ij = sum_kl B_ijkl X_kl.
    Differentiable in (A, B) by the implicit function theorem (the solver runs detached, in float64 on the CPU)."""
    k = A.shape[0]
    basis = symmetric_basis(k)
    to_coords = basis.to(A).T
    A_coords, B_coords = to_coords @ A.flatten(), to_coords @ B.reshape(k * k, k * k) @ to_coords.T
    X, sensitivity = (t.to(A) for t in solve_sdcp(A_coords.detach().double().cpu(), B_coords.detach().double().cpu(), basis))
    slack = A_coords + B_coords @ X
    X = X + sensitivity @ (slack - slack.detach())  # the same value, with the solution's derivative
    return (to_coords.T @ X).reshape(k, k)
