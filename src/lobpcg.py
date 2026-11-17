"""LOBPCG (`torch_lobpcg`, copied and translated into PyTorch from Google's JAX implementation:
https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/linalg.py#L37-L105), the Hessian eigenvalues it computes
(`hessian_eigs`) and the dict <-> flat translation (`flatten` / `unflatten`) of the flat-vector black boxes (it and `minres`)."""

from typing import Callable, Mapping

import torch
from torch import Tensor

from src.utils import HVP


def torch_lobpcg(
    A: Callable[[Tensor], Tensor],
    X: Tensor,
    max_iter=200,  # SGD: ~1-5 for positive eigs, ~10-20 for negative eigs
    rtol=0.03,  # barely any difference from 1e-3 to 1e-1
) -> tuple[Tensor, Tensor, int]:
    """Finds the top k eigenvalues/eigenvectors of the n x n linear operator A.

    Args:
      A (Callable: Array -> Array): function representing an [n x n] linear operator.
        Passing in an [n x r] input should return an [n x r] output where A has
        been evaluated on each column of the input.
      X (Array): an initial guess for the top k eigenvectors, of shape [n x k].  This is
        how the function learns what k is.
      max_iter (int): the maximum number of LOBPCG iterations to run
      rtol (float): column i has converged when its residual ||A x_i - theta_i x_i|| is below
        rtol * (||A x_i|| + theta_i), so each theta_i is within about that of a true eigenvalue

    Returns:
      top eigenvalues [k]
      top eigenvectors [n x k]
      number of iterations run (max_iter if it didn't converge)
    """
    n, k = X.shape

    if n < 4 * k:
        A_mat = A(torch.eye(n, dtype=X.dtype, device=X.device))
        evals, evecs = _eigh_ascending(A_mat)
        return evals[:k], evecs[:, :k], 0

    X = _orthonormalize(X)
    P = _extend_basis(X, X.shape[1])
    AX = A(X)
    theta = torch.sum(X * AX, dim=0, keepdim=True)
    R = AX - theta * X
    converged = 0
    n_iter = 0
    while converged < k and n_iter < max_iter:
        n_iter += 1
        R = _project_out(torch.cat((X, P), dim=1), R)
        XPR = torch.cat((X, P, R), dim=1)
        A_XPR = A(XPR)
        theta, Q = _eigh_ascending(XPR.T @ A_XPR)
        theta = theta[:k]
        B = Q[:, :k]
        normB = torch.linalg.norm(B, ord=2, dim=0, keepdim=True)
        B /= normB
        X = XPR @ B
        normX = torch.linalg.norm(X, ord=2, dim=0, keepdim=True)
        X /= normX
        q, _ = torch.linalg.qr(Q[:k, k:].T)
        diff_rayleigh_ortho = Q[:, k:] @ q
        P = XPR @ diff_rayleigh_ortho
        normP = torch.linalg.norm(P, ord=2, dim=0, keepdim=True)
        P /= torch.where(normP == 0, 1.0, normP)
        AX = A_XPR @ B / normX  # = A(X)
        R = AX - theta[None, :] * X
        resid_norms = torch.linalg.norm(R, ord=2, dim=0)
        # |theta| (JAX: theta): on an indefinite operator a negative Ritz value would make the bound ~0
        res_converged = resid_norms < rtol * (torch.linalg.norm(AX, ord=2, dim=0) + theta.abs())
        converged = torch.sum(res_converged)
    return theta, X, n_iter


def _project_out(basis, U):
    for _ in range(2):
        U -= basis @ (basis.T @ U)
        U = _orthonormalize(U)
    for _ in range(2):
        U -= basis @ (basis.T @ U)
    normU = torch.linalg.norm(U, ord=2, dim=0, keepdim=True)
    U *= normU >= 0.99

    return U

def _orthonormalize(basis):
    for _ in range(2):
        basis = _svqb(basis)
    return basis


def _eigh_ascending(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(Despite the name, kept from JAX) eigenvalues in descending order."""
    w, V = torch.linalg.eigh(A)
    idx = torch.argsort(-w)
    return w[idx], V[:, idx]


def _svqb(X):
    norms = torch.linalg.norm(X, ord=2, dim=0, keepdim=True)
    X /= torch.where(norms == 0, 1.0, norms)
    inner = X.T @ X
    w, V = _eigh_ascending(inner)
    tau = torch.finfo(X.dtype).eps * w[0]
    padded = torch.maximum(w, tau)
    sqrted = torch.where(tau > 0, padded, 1.0) ** (-0.5)
    scaledV = V * sqrted[None, :]
    orthoX = X @ scaledV
    keep = ((w > tau) * (torch.diag(inner) > 0.0))[None, :]
    orthoX *= keep.type(orthoX.dtype)
    norms = torch.linalg.norm(orthoX, ord=2, dim=0, keepdim=True)
    keep *= (norms > 0.0).type(keep.dtype)
    orthoX /= torch.where(keep, norms, 1.0)
    return orthoX


def _extend_basis(X, m):
    n, k = X.shape
    Xupper, Xlower = X[:k], X[k:]
    u, s, vt = torch.linalg.svd(Xupper)
    y = torch.concatenate([Xupper + u @ vt, Xlower], dim=0)
    other = torch.concatenate(
        [
            torch.eye(m, dtype=X.dtype, device=X.device),
            torch.zeros((n - k - m, m), dtype=X.dtype, device=X.device),
        ],
        dim=0,
    )
    w = y @ (vt.T * ((2 * (1 + s)) ** (-1 / 2))[None, :])
    h = -2 * torch.linalg.multi_dot([w, w[k:, :].T, other])
    h[k:] += other
    return h


def flatten(d: dict[str, Tensor], batched: bool = True) -> Tensor:
    """{name: [m, *shape]} -> [#params, m] (the batch as columns); not batched: {name: [*shape]} -> [#params]"""
    return torch.cat([v.flatten(1) for v in d.values()], dim=1).T if batched else torch.cat([v.flatten() for v in d.values()])


def unflatten(V: Tensor, shapes: Mapping[str, torch.Size]) -> dict[str, Tensor]:
    """Inverse of `flatten`: [#params, m] -> {name: [m, *shapes[name]]}, [#params] -> {name: [*shapes[name]]}"""
    return {
        name: v.T.reshape(-1, *shape) if V.ndim == 2 else v.reshape(shape)
        for v, (name, shape) in zip(V.split([shape.numel() for shape in shapes.values()]), shapes.items())
    }


def hessian_eigs(
    hvp_fn: HVP, eigvecs: dict[str, Tensor],
    precond: Mapping[str, Tensor | float] | None = None, sign: float = 1, batched: bool = True, rtol: float = 0.03,
) -> tuple[Tensor, dict[str, Tensor], int]:
    """The top (sign -1: most negative) eigenvalues of sqrt(P) H sqrt(P) by `torch_lobpcg`.
    eigvecs {name: [k, *shape]} -> (eigenvalues [k], eigenvectors like `eigvecs`, LOBPCG iterations)
    Not batched: a single start vector {name: [*shape]} -> (eigenvalue [], eigenvector like it, iterations)
    """
    shapes = {name: v.shape[1:] if batched else v.shape for name, v in eigvecs.items()}
    sqrt_p = None if precond is None else flatten({
        name: torch.as_tensor(precond[name], dtype=v.dtype, device=v.device).expand(shape)
        for (name, shape), v in zip(shapes.items(), eigvecs.values())
    }, batched=False).sqrt()[:, None]

    def A(V: Tensor) -> Tensor:  # [#params, m] -> [#params, m] (LOBPCG uses m = k and 3k)
        if sqrt_p is not None:
            V = sqrt_p * V
        HV = flatten(hvp_fn(unflatten(V, shapes)))
        return sign * (HV if sqrt_p is None else sqrt_p * HV)

    X = flatten(eigvecs, batched)
    eigvals, V, n_iter = torch_lobpcg(A, X if batched else X[:, None], rtol=rtol)
    return sign * (eigvals if batched else eigvals[0]), unflatten(V if batched else V[:, 0], shapes), n_iter
