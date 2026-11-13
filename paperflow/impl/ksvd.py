import torch
from typing import Optional, Tuple, List

# ---------------------------------------------------------------------
# Utility: normalize columns, set fast kernels, device helpers
# ---------------------------------------------------------------------
def _normalize_columns_(D: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    nrm = torch.linalg.norm(D, dim=0, keepdim=True).clamp_min(eps)
    D /= nrm
    return D

def _maybe_enable_fast_kernels():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------
# Sparse coding (batch, vectorized):
#  - pick top-s atoms by |D^T y|
#  - refit via (D_S^T D_S) x = D_S^T y (batched Cholesky)
#  Memory-safe via column batching.
# ---------------------------------------------------------------------
@torch.no_grad()
def batch_threshold_code(
    Y: torch.Tensor,          # (m, N)
    D: torch.Tensor,          # (m, K) (columns normalized)
    s: int,                   # per-sample sparsity
    batch_size: int = 8192,
    ridge: float = 1e-6       # tiny Tikhonov for stability
) -> torch.Tensor:            # returns X (K, N)
    m, N = Y.shape
    K = D.shape[1]
    device = Y.device
    dtype = Y.dtype

    # Precompute Gram once.
    G = (D.T @ D).to(dtype)                       # (K, K)
    X = torch.zeros((K, N), device=device, dtype=dtype)

    I_s_cache = None  # reuse identity per inner batch

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Yb = Y[:, start:end]                      # (m, B)
        B = Yb.shape[1]

        DtYb = D.T @ Yb                           # (K, B)
        idx = torch.topk(DtYb.abs(), k=s, dim=0, largest=True, sorted=False).indices  # (s, B)
        S = idx.T.contiguous()                    # (B, s)

        # Build batched G_SS: (B, s, s)
        # Step1: select rows for each sample -> (B, s, K)
        G_rows = G[S]                             
        # Step2: select columns using the same indices per batch
        G_SS = torch.gather(G_rows, 2, S.unsqueeze(1).expand(B, s, s))

        # Right-hand sides b = D_S^T y (B, s)
        b = DtYb.gather(0, S.T).T.contiguous()    # (B, s)

        # Cholesky solves (G_SS + lambda I) x = b
        if I_s_cache is None or I_s_cache.shape[-1] != s:
            I_s_cache = torch.eye(s, device=device, dtype=dtype).unsqueeze(0)
        L = torch.linalg.cholesky(G_SS + ridge * I_s_cache)
        # cholesky_solve expects (..., s, s) and (..., s, k). Here k=1.
        z = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)  # (B, s)

        # Scatter back to X[:, start:end]
        Xb = torch.zeros((K, B), device=device, dtype=dtype)
        Xb.scatter_(0, S.T, z.T)
        X[:, start:end] = Xb

    return X


# ---------------------------------------------------------------------
# Rank-1 dictionary atom update (K–SVD core) with incremental residual
# Choose solver: "pca" (fast, default), "svd" (exact), or "power" (few iters).
# ---------------------------------------------------------------------
def _top_rank1(
    M: torch.Tensor,
    solver: str = "pca",
    power_iters: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-1 SVD of M ≈ u*s*v^T.
    Returns (u, s_v) where u is (m,), s_v is (n,) such that M ≈ u[:,None] @ s_v[None,:]
    i.e., s_v == s * v.
    """
    m, n = M.shape
    if solver == "svd":
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        u = U[:, 0]
        s_v = S[0] * Vh[0, :]
        return u, s_v
    elif solver == "pca":
        # Uses torch.pca_lowrank; with center=False behaves like rank-1 SVD.
        q = int(min(8, m, n))  # oversample
        U, S, V = torch.pca_lowrank(M, q=q, center=False, niter=3)
        u = U[:, 0]
        s_v = S[0] * V[:, 0]
        return u, s_v
    elif solver == "power":
        # Power iteration on M M^T to get u, then s_v = u^T M
        u = torch.randn(m, device=M.device, dtype=M.dtype)
        u = u / (u.norm() + 1e-12)
        for _ in range(power_iters):
            u = M @ (M.T @ u)
            u = u / (u.norm() + 1e-12)
        s_v = u @ M  # (n,)
        return u, s_v
    else:
        raise ValueError("solver must be one of {'pca','svd','power'}")


@torch.no_grad()
def ksvd(
    Y: torch.Tensor,                # (m, N) data matrix (columns are signals)
    n_components: int,              # K
    n_iter: int = 10,
    sparsity: int = 8,
    D_init: Optional[torch.Tensor] = None,  # (m, K)
    C_init: Optional[torch.Tensor] = None,  # (K, N)
    batch_size: int = 8192,
    ridge: float = 1e-6,
    svd_solver: str = "pca",        # "pca" | "svd" | "power"
    power_iters: int = 3,
    reinit_on_empty: bool = True,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Fast K–SVD with batch sparse coding and incremental residual updates.

    Returns:
        D: (m, K) learned dictionary (unit-norm columns)
        X: (K, N) sparse codes (s-sparse)
        errs: list of Frobenius reconstruction errors per iteration
    """
    _maybe_enable_fast_kernels()
    device, dtype = Y.device, Y.dtype
    m, N = Y.shape
    K = n_components
    assert sparsity > 0 and sparsity <= min(K, m), "Invalid sparsity."
    assert K > 0 and K <= max(m * 4, 32768), "Suspicious dictionary size."

    # Initialize dictionary
    if D_init is not None:
        D = D_init.to(device=device, dtype=dtype).clone()
        _normalize_columns_(D)
        if D.shape != (m, K):
            raise ValueError("D_init has wrong shape.")
    else:
        # Pick K random columns of Y (with replacement) and normalize
        if N >= K:
            perm = torch.randint(0, N, (K,), device=device)
            D = Y[:, perm].clone()
        else:
            D = torch.randn((m, K), device=device, dtype=dtype)
        _normalize_columns_(D)

    errs: List[float] = []

    for it in range(n_iter):
        # -------- Sparse coding --------
        if C_init is not None and it == 0: # in case we performed some rough estimation of coefficients
            X = C_init
        else:
            X = batch_threshold_code(Y, D, s=sparsity, batch_size=batch_size, ridge=ridge)
        # -------- Residual --------
        R = Y - D @ X  # (m, N)
        err = float(torch.linalg.norm(R).item())
        errs.append(err)
        if verbose:
            print(f"[{it+1:02d}/{n_iter}] ||Y - D X||_F = {err:.4e}")

        # -------- Dictionary update (atom-by-atom) --------
        for k in range(K):
            xk = X[k, :]  # (N,)
            omega = torch.nonzero(xk, as_tuple=False).squeeze(1)  # indices where atom k is used
            if omega.numel() == 0:
                if reinit_on_empty:
                    # Reinitialize atom k to largest residual column
                    col_norms = torch.linalg.norm(R, dim=0)
                    j = int(torch.argmax(col_norms).item())
                    new_atom = R[:, j]
                    nrm = float(new_atom.norm().item())
                    if nrm < 1e-12:
                        # Fallback to random
                        new_atom = torch.randn(m, device=device, dtype=dtype)
                        nrm = float(new_atom.norm().item())
                    D[:, k] = new_atom / (nrm + 1e-12)
                    X[k, :] = 0.0
                    # Put a single nonzero coefficient to capture that column
                    X[k, j] = nrm
                    # Update residual incrementally for that column
                    R[:, j] -= D[:, k] * X[k, j]
                continue

            # Add back current contribution of atom k on its support to form E_k
            Rk = R[:, omega] + D[:, k].unsqueeze(1) * xk[omega].unsqueeze(0)  # (m, |omega|)

            # Rank-1 update via top singular pair
            u, s_v = _top_rank1(Rk, solver=svd_solver, power_iters=power_iters)
            # Normalize u to unit norm (pca/svd already unit, but be safe)
            unrm = float(u.norm().item())
            if unrm < 1e-12:
                # Degenerate; skip update
                continue
            u = u / unrm
            # Update dictionary atom and its coefficients on support
            D[:, k] = u
            X[k, omega] = s_v  # s * v

            # Recompute R on omega incrementally: R_new = Rk - u @ (s_v)^T
            R[:, omega] = Rk - u.unsqueeze(1) * s_v.unsqueeze(0)

        # (Optional) renormalize to avoid drift
        _normalize_columns_(D)

    # Final residual report
    if verbose:
        R = Y - D @ X
        err = float(torch.linalg.norm(R).item())
        print(f"[done] final ||Y - D X||_F = {err:.4e}")

    return D, X, errs

