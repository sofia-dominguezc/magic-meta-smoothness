import torch
from typing import Callable, Mapping

"""Native PyTorch implementation of the LOBPCG algorithm for eigenvalue computation.

This code was copied (and translated into PyTorch) from Google's JAX implementation:
https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/linalg.py#L37-L105
"""

def torch_lobpcg(
    A: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor, max_iter=100, rtol=1e-2
) -> tuple[torch.Tensor, torch.Tensor, int]:
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

def _rayleigh_ritz_orth(A, S):
    SAS = S.T @ A(S)
    return _eigh_ascending(SAS)


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


def top_hessian_eigs(
    grads: dict[str, torch.Tensor], params: dict[str, torch.Tensor], k: int,
    eigvecs: torch.Tensor | None = None, precond: Mapping[str, torch.Tensor | float] | None = None, seed: int = 0,
    negative: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Top k eigenvalues of the Hessian H (w.r.t. `params`) of the loss whose gradient is `grads`
    (computed with create_graph=True), by LOBPCG.
    Args:
    - eigvecs: [#params, k] to warm-start LOBPCG (None: random with `seed`)
    - precond: per-element step size P (`OptimState.precond`; None: 1), each broadcastable to its param: then the
      eigenvalues are those of sqrt(P) H sqrt(P), which govern the stability of the step
    - negative: the k most negative eigenvalues instead (LOBPCG on -H), most negative first
    Returns: (eigenvalues [k], eigenvectors [#params, k] to warm-start the next call, LOBPCG iterations).
    """
    sizes, sqrt_p, sign = [p.numel() for p in params.values()], None, -1.0 if negative else 1.0
    if precond is not None:
        sqrt_p = torch.cat([
            torch.as_tensor(m, dtype=p.dtype, device=p.device).expand_as(p).flatten()
            for m, p in zip(precond.values(), params.values())
        ]).sqrt()[:, None]

    def hvp(V: torch.Tensor) -> torch.Tensor:  # [#params, m] -> [#params, m] (LOBPCG uses m = k and 3k)
        if sqrt_p is not None:
            V = sqrt_p * V
        V_list = [v.T.reshape(-1, *p.shape) for v, p in zip(V.split(sizes), params.values())]
        HV_dict = torch.autograd.grad(
            list(grads.values()), params, V_list, retain_graph=True, is_grads_batched=True
        )
        HV = torch.cat([hv.reshape(len(hv), -1) for hv in HV_dict.values()], dim=1).T
        return sign * (HV if sqrt_p is None else sqrt_p * HV)

    if eigvecs is None:
        g = torch.Generator().manual_seed(seed)
        eigvecs = torch.randn(sum(sizes), k, generator=g).to(next(iter(params.values())))
    eigvals, eigvecs, n_iter = torch_lobpcg(hvp, eigvecs)
    return sign * eigvals, eigvecs, n_iter
