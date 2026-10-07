import torch
from typing import Callable

"""Native PyTorch implementation of the LOBPCG algorithm for eigenvalue computation.

This code was copied (and translated into PyTorch) from Google's JAX implementation:
https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/linalg.py#L37-L105
"""

def torch_lobpcg(
    A: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor, max_iter=100, tol=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finds the top k eigenvalues/eigenvectors of the n x n linear operator A.

    Args:
      A (Callable: Array -> Array): function representing an [n x n] linear operator. 
        Passing in an [n x r] input should return an [n x r] output where A has
        been evaluated on each column of the input.
      X (Array): an initial guess for the top k eigenvectors, of shape [n x k].  This is
        how the function learns what k is.
      max_iter (int): the maximum number of LOBPCG iterations to run
      tol (float): the numerical tolerance

    Returns:
      top eigenvalues [k]
      top eigenvectors [n x k]
    """
    n, k = X.shape

    if n < 4 * k:
        A_mat = A(torch.eye(n, dtype=X.dtype, device=X.device))
        evals, evecs = _eigh_ascending(A_mat)
        return evals[:k], evecs[:, :k]

    if tol is None:
        tol = torch.finfo(X.dtype).eps

    X = _orthonormalize(X)
    P = _extend_basis(X, X.shape[1])
    AX = A(X)
    theta = torch.sum(X * AX, dim=0, keepdim=True)
    R = AX - theta * X
    converged = 0
    for i in range(max_iter):
        if converged >= k:
            break
        R = _project_out(torch.cat((X, P), dim=1), R)
        XPR = torch.cat((X, P, R), dim=1)
        theta, Q = _rayleigh_ritz_orth(A, XPR)
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
        AX = A(X)
        R = AX - theta[None, :] * X
        resid_norms = torch.linalg.norm(R, ord=2, dim=0)
        reltol = 10 * n * (torch.linalg.norm(AX, ord=2, dim=0) + theta)
        res_converged = resid_norms < tol * reltol
        converged = torch.sum(res_converged)
    return theta, X


def _project_out(basis, U):
    for _ in range(2):
        U -= basis @ (basis.T @ U)
        U = _orthonormalize(U)
    for _ in range(2):
        U -= basis @ (basis.T @ U)
    normU = torch.linalg.norm(U, ord=2, dim=0, keepdim=True)
    U *= normU >= 0.99

    return U

def _rayleigh_ritz_orth(A, S):
    SAS = S.T @ A(S)
    return _eigh_ascending(SAS)


def _orthonormalize(basis):
    for _ in range(2):
        basis = _svqb(basis)
    return basis


def _eigh_ascending(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        w, V = torch.linalg.eigh(A)
    except:
        print("uh oh")
        raise Exception("eigh failed")
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
