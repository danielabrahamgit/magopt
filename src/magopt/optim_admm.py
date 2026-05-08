import torch
import cvxpy as cp

from torch import nn
from torch.backends import cpu
from tqdm import tqdm
from typing import Optional, Callable, Union
from einops import rearrange

# Constants
lamda_stability = 1e-6  # for numerical stability in x-update

def detach_state(*args):
    return [x.detach() for x in args]

def detach_state_cpu(*args):
    return [x.detach().cpu() for x in args]

@torch.no_grad()
def boyd_update(r_norm, s_norm, rho, u, tau=2.0, mu=10.0, rho_min=1e-6, rho_max=1e6):
    """
    If ||r|| > mu*||s||:   rho <- min(tau*rho, rho_max),  u <- u * (rho/rho_new)
    elif ||s|| > mu*||r||: rho <- max(rho/tau, rho_min),  u <- u * (rho/rho_new)
    else: leave unchanged.
    (u is the *scaled* dual for this block.)
    """
    rho_new = rho
    if r_norm > mu * s_norm and rho < rho_max:
        rho_new = min(rho * tau, rho_max)
    elif s_norm > mu * r_norm and rho > rho_min:
        rho_new = max(rho / tau, rho_min)

    if rho_new != rho:
        u = u * (rho / rho_new)  # keep unscaled dual constant
    return rho_new, u

def proj_ellipsoid(z: torch.Tensor, 
                   L: torch.Tensor, 
                   L_max: float,
                   tol: float = 1e-8, 
                   max_iter: int = 100) -> torch.Tensor:
    """
    Solve:  minimize_s  ||z - s||_2^2
            subject to  s^T L s <= L_max

    Parameters
    ----------
    z : torch.Tensor, shape (N,)
        Input vector.
    L : torch.Tensor, shape (N, N)
        Symmetric positive semidefinite matrix defining the quadratic form.
    L_max : float
        Upper bound for s^T L s.
    tol : float
        Tolerance for bisection.
    max_iter : int
        Maximum number of bisection iterations.

    Returns
    -------
    s : torch.Tensor, shape (N,)
        Projection of z onto {s: s^T L s <= L_max}.
    """

    # Ensure proper shape and dtype
    assert z.ndim == 1 and L.ndim == 2 and L.shape[0] == L.shape[1] == z.shape[0]
    N = z.shape[0]

    # Compute L^(1/2) decomposition (eig works even if L is PSD)
    eigvals, Q = torch.linalg.eigh(L)
    eigvals = eigvals.clamp(min=0)  # ensure no negative values due to numerical noise

    # Check feasibility: if z already satisfies constraint, return z
    val = (z @ (L @ z)).item()
    if val <= L_max + 1e-12:
        return z.clone()

    # Transform to eigenbasis
    a = Q.T @ z

    # Define function φ(λ) = sum_i [ (λ_i * a_i^2) / (1 + λ * λ_i)^2 ] - L_max
    def phi(lam):
        denom = (1.0 + lam * eigvals)
        return torch.sum(eigvals * (a ** 2) / (denom ** 2)) - L_max

    # Bisection over λ >= 0 to satisfy φ(λ) = 0
    lam_low = torch.tensor(0.0, dtype=z.dtype, device=z.device)
    lam_high = torch.tensor(1.0, dtype=z.dtype, device=z.device)
    # Increase lam_high until φ(lam_high) < 0
    while phi(lam_high) > 0:
        lam_high *= 2
        if lam_high > 1e10:
            break

    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_low + lam_high)
        val_mid = phi(lam_mid)
        if torch.abs(val_mid) < tol:
            break
        if val_mid > 0:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    lam = lam_mid
    y = a / (1.0 + lam * eigvals)
    s = Q @ y
    return s

def proj_l2_ball(z, tau) -> torch.Tensor:
    """
    Project each row of Z onto an l2-ball:
        Z[i] <- argmin_u ||u - Z[i]||_2  s.t. ||u||_2 <= tau
    """
    norms = torch.linalg.norm(z, dim=1, keepdim=True)  # (K,1)
    eps = torch.finfo(z.dtype).eps
    scale = torch.clamp(tau / (norms + eps), max=1.0)
    return z * scale

def solve_epigraph_group_l2(y, lam, max_iter=20, tol=1e-6):
    """
    Differentiable PyTorch solver for
        min_{z,t} t + lam * sum_k ||y_k - z_k||^2
        s.t.     ||z_k|| <= t
    Returns (z, t).
    Works on GPU, differentiable w.r.t. y.
    """
    # group norms
    r = torch.linalg.norm(y, dim=tuple(range(1, y.ndim)))  # (K,)
    K = r.numel()

    # initialization: t0 between 0 and max norm
    t = r.mean().detach()  # safe init

    for _ in range(max_iter):
        diff = (r - t).clamp(min=0)     # (||y|| - t)_+
        grad = 1 - 2 * lam * diff.sum()
        hess = 2 * lam * (diff > 0).sum(dtype=y.dtype)
        step = grad / (hess + 1e-12)
        t_new = (t - step).clamp(min=0)
        if torch.allclose(t, t_new, rtol=0, atol=tol):
            t = t_new
            break
        t = t_new

    # projection step
    scales = torch.clamp(t / (r + 1e-12), max=1.0)
    while scales.ndim < y.ndim:
        scales = scales.unsqueeze(-1)
    z = y * scales
    return z, t

def e_block_update_exact(v: torch.Tensor,
                         lambdaE: float | torch.Tensor,
                         rhoE: float | torch.Tensor,
                         e_fixed: float | torch.Tensor | None = None,
                         vec_dim: int = -1,
                         eps: float = 1e-12) -> tuple[torch.Tensor, float]:
    """
    Exact (non-differentiable) E-block ADMM update.

    Solves:
        min_{z_k, e>=0} lambdaE*e + (rhoE/2) * sum_k ||z_k - v_k||^2
        s.t. ||z_k|| <= e  for all k

    where v_k are provided as `v`.

    Args
    ----
    v : Tensor
        Stacked v_k. Shape: (K, ..., D) where D is the vector dimension over which ||.||_2 is computed.
        The "k index" is assumed to be dimension 0 (K).
    lambdaE, rhoE : float or Tensor
        Scalars.
    e_fixed : float or Tensor or None
        If not None, hard-sets e = e_fixed and only projects z_k onto the ball of radius e.
    vec_dim : int
        Dimension of the vector entries (default: last dim).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    z : Tensor
        Same shape as v, with each v_k projected onto the l2 ball radius e.
    e : Tensor (scalar)
        Optimal (or fixed) Emax.
    """
    if v.numel() == 0:
        raise ValueError("v must be non-empty.")
    if v.shape[0] == 0:
        raise ValueError("First dimension of v (K) must be > 0.")

    device, dtype = v.device, v.dtype
    lam = torch.as_tensor(lambdaE, device=device, dtype=dtype)
    rho = torch.as_tensor(rhoE, device=device, dtype=dtype)

    # Compute a_k = ||v_k||_2
    # Assume k is dim 0; compute norm over vec_dim.
    # If vec_dim is negative, it refers to indexing in v; keep as-is.
    a = torch.linalg.vector_norm(v, ord=2, dim=vec_dim)  # shape: (K, ...)

    # This solver assumes one scalar e shared across *all* k and all other batch dims.
    # So we flatten all a entries into one list of radii.
    a_flat = a.reshape(-1)

    # Hard-set e case
    if e_fixed is not None:
        e = torch.as_tensor(e_fixed, device=device, dtype=dtype).clamp_min(0.0)
        # project each v elementwise by scaling factor min(1, e/||v||)
        v_norm = torch.linalg.vector_norm(v, ord=2, dim=vec_dim, keepdim=True).clamp_min(eps)
        scale = torch.clamp(e / v_norm, max=1.0)
        z = v * scale
        return z, e

    # If lambdaE <= 0, objective would push e -> +inf (not your use case).
    # If lambdaE is very small, e tends toward max ||v_k||.
    # The optimality is: e* = argmin_{e>=0} lam e + (rho/2) sum (max(0, a_k - e))^2

    # Sort descending: a_(1) >= a_(2) >= ... >= a_(N)
    a_sorted, _ = torch.sort(a_flat, descending=True)
    N = a_sorted.numel()

    # Prefix sums S_m = sum_{i=1}^m a_(i)
    S = torch.cumsum(a_sorted, dim=0)

    # Candidates e_m = (S_m - lam/rho)/m for m=1..N
    target = lam / (rho + eps)
    m = torch.arange(1, N + 1, device=device, dtype=dtype)
    e_cand = (S - target) / m

    # Enforce e >= 0
    e_cand = torch.clamp(e_cand, min=0.0)

    # Valid interval condition for descending sort:
    # a_m >= e_m >= a_{m+1}, with a_{N+1} := 0
    a_next = torch.empty_like(a_sorted)
    a_next[:-1] = a_sorted[1:]
    a_next[-1] = torch.zeros((), device=device, dtype=dtype)

    valid = (e_cand <= a_sorted + 1e-14) & (e_cand >= a_next - 1e-14)

    if valid.any():
        idx = torch.nonzero(valid, as_tuple=False)[0, 0]
        e = e_cand[idx]
    else:
        # Fallback (rare numerical corner): if all invalid, choose e=0
        e = torch.zeros((), device=device, dtype=dtype)

    # Project each v_k onto l2-ball radius e: z_k = v_k * min(1, e/||v_k||)
    v_norm = torch.linalg.vector_norm(v, ord=2, dim=vec_dim, keepdim=True).clamp_min(eps)
    scale = torch.clamp(e / v_norm, max=1.0)
    z = v * scale

    return z, e

def g_block_update_diff(q: torch.Tensor,
                        lambdaG: torch.Tensor | float,
                        rhoG: torch.Tensor | float,
                        tau: float = 1e-3,
                        newton_iters: int = 25,
                        eps: float = 1e-12) -> tuple[torch.Tensor, float]:
    """
    Differentiable solver for the G-block update (scalar Gmin = gamma).

    Solves a smooth approximation of:
        min_{g,gamma} -lambdaG*gamma + (rhoG/2)||g-q||^2   s.t. g >= gamma
    via:
        hinge(t)=max(0,t)  ~  tau*softplus(t/tau)

    Returns:
        g     : same shape as q
        gamma : scalar tensor (broadcastable)

    Notes:
      - As tau -> 0, this approaches the exact solution, but gradients can become sharp.
      - newton_iters is unrolled => differentiable w.r.t. q, lambdaG, rhoG.
    """
    if q.ndim < 1:
        raise ValueError("q must be at least 1D")

    device, dtype = q.device, q.dtype
    lambdaG_t = torch.as_tensor(lambdaG, device=device, dtype=dtype)
    rhoG_t = torch.as_tensor(rhoG, device=device, dtype=dtype)

    # target = lambdaG / rhoG (scalar)
    target = lambdaG_t / (rhoG_t + eps)

    # Smooth hinge: relu(t) ~ tau*softplus(t/tau)
    # Define h(gamma) = sum_i tau*softplus((gamma - q_i)/tau) - target = 0
    # h'(gamma) = sum_i sigmoid((gamma - q_i)/tau)
    #
    # Initialize gamma near max(q). If target>0, gamma will usually be >= max(q).
    q_max = q.max()
    # A mild upward bias helps Newton converge quickly in typical cases.
    gamma = q_max + target / (q.numel() + eps)

    for _ in range(newton_iters):
        s = (gamma - q) / tau
        hinge_smooth = tau * F.softplus(s)             # ~ max(0, gamma - q)
        h = hinge_smooth.sum() - target

        dh = torch.sigmoid(s).sum() / (1.0 + 0.0)      # derivative wrt gamma
        dh = dh.clamp_min(eps)

        gamma = gamma - h / dh

    # Smooth max: max(q, gamma) ~ q + tau*softplus((gamma - q)/tau)
    g = q + tau * F.softplus((gamma - q) / tau)

    return g, gamma

def g_block_update_exact(q: torch.Tensor,
                         lambdaG: float | torch.Tensor,
                         rhoG: float | torch.Tensor,
                         eps: float = 1e-12) -> tuple[torch.Tensor, float]:
    """
    Exact (non-differentiable) solver for the G-block update with scalar Gmin = gamma.

    Solves:
        min_{g,gamma} -lambdaG*gamma + (rhoG/2)||g-q||^2  s.t. g >= gamma

    Returns:
        g     : same shape as q
        gamma : scalar tensor (same dtype/device)

    Notes:
        - Uses the closed-form sort + prefix-sum method.
        - Marked @torch.no_grad() since it's intended non-differentiable.
        - Works for any q shape; treats q as a flat vector of constraints.
    """
    if q.numel() == 0:
        raise ValueError("q must be non-empty")

    device, dtype = q.device, q.dtype
    lam = torch.as_tensor(lambdaG, device=device, dtype=dtype)
    rho = torch.as_tensor(rhoG, device=device, dtype=dtype)
    target = lam / (rho + eps)  # lambdaG / rhoG

    q_flat = q.reshape(-1)

    # If target == 0, best is gamma = min(q)?? Let's see:
    # Condition sum max(0, gamma - q_i) = 0 => gamma <= min(q).
    # Objective prefers large gamma, so choose gamma = min(q).
    if float(target.item()) <= 0.0:
        gamma = q_flat.min()
        g = torch.maximum(q, gamma)
        return g, gamma

    # Sort ascending: b_1 <= ... <= b_n
    b, _ = torch.sort(q_flat)
    n = b.numel()

    # Prefix sums T_m = sum_{i=1}^m b_i
    T = torch.cumsum(b, dim=0)

    # Candidate gamma_m = (T_m + target) / m  (m is 1-indexed)
    m = torch.arange(1, n + 1, device=device, dtype=dtype)
    gamma_candidates = (T + target) / m

    # We need b_m <= gamma_m <= b_{m+1}, with b_{n+1}=+inf
    # Create b_{m+1} by shifting left and appending +inf
    b_next = torch.empty_like(b)
    b_next[:-1] = b[1:]
    b_next[-1] = torch.tensor(float("inf"), device=device, dtype=dtype)

    valid = (gamma_candidates >= b) & (gamma_candidates <= b_next)

    if valid.any():
        # pick the first valid m (smallest m) — it's the correct interval
        idx = torch.nonzero(valid, as_tuple=False)[0, 0]
        gamma = gamma_candidates[idx]
    else:
        # Numerical fallback: if no interval matched due to precision,
        # clamp to [b_1, +inf) using the last candidate.
        gamma = gamma_candidates[-1].clamp_min(b[0])

    # g = max(q, gamma)
    gamma_b = gamma.reshape(*([1] * q.ndim))  # broadcast scalar to q
    g = torch.maximum(q, gamma_b)

    return g, gamma

def g_block_update_band_exact(q: torch.Tensor,
                              lambdaG: float | torch.Tensor,
                              rhoG: float | torch.Tensor,
                              linearity_pcnt: float,
                              gamma_min: float | None = None,   # e.g. 0.0 if you need Gmin >= 0
                              max_iters: int = 80,
                              tol: float = 1e-10) -> tuple[torch.Tensor, float]:
    """
    Exact (non-differentiable) G-block update for constraint:
        gamma <= g <= (1+linearity_pcnt)*gamma,  where g = Gx

    Solves:
        min_{g,gamma} -lambdaG*gamma + (rhoG/2)||g-q||^2
        s.t. gamma <= g_i <= (1+linearity_pcnt)*gamma  (for all i)

    Returns:
        g     : same shape as q
        gamma : scalar tensor
    """
    if linearity_pcnt < -1.0:
        raise ValueError("Need 1+linearity_pcnt >= 0 for convex 'band' constraint.")
    if q.numel() == 0:
        raise ValueError("q must be non-empty.")

    device, dtype = q.device, q.dtype
    lam = torch.as_tensor(lambdaG, device=device, dtype=dtype)
    rho = torch.as_tensor(rhoG, device=device, dtype=dtype)
    a = torch.as_tensor(1.0 + linearity_pcnt, device=device, dtype=dtype)

    qf = q.reshape(-1)

    # derivative of f(gamma):
    # f(gamma) = -lam*gamma + (rho/2) sum_i (clip(q_i, [gamma, a*gamma]) - q_i)^2
    #
    # Let I_low = {i: q_i < gamma} -> clip = gamma -> contrib (gamma-q_i)^2
    # Let I_hi  = {i: q_i > a*gamma} -> clip = a*gamma -> contrib (a*gamma-q_i)^2
    #
    # f'(gamma) = -lam + rho * [ sum_{i in I_low} (gamma - q_i) + a * sum_{i in I_hi} (a*gamma - q_i) ]
    #
    def fprime(gamma: torch.Tensor) -> torch.Tensor:
        low = qf < gamma
        hi = qf > a * gamma
        # sums over selected sets
        term_low = (gamma - qf[low]).sum()
        term_hi = (a * gamma - qf[hi]).sum()
        return -lam + rho * (term_low + a * term_hi)

    # Choose a bracket [lo, hi] with f'(lo) <= 0 <= f'(hi)
    # f' is monotone increasing (convex 1D), so bisection works.
    #
    # A safe starting point is around min/max of q scaled.
    qmin = qf.min()
    qmax = qf.max()

    # Initial bracket heuristics:
    # For very small gamma, many points are in "hi" (if a*gamma << q), making f'(gamma) very negative.
    # For very large gamma, many are in "low", making f'(gamma) very positive.
    lo = qmin / a - (qmax - qmin + 1.0)  # conservative
    hi = qmax + (qmax - qmin + 1.0)

    if gamma_min is not None:
        lo = torch.maximum(torch.as_tensor(gamma_min, device=device, dtype=dtype), torch.as_tensor(lo, device=device, dtype=dtype))
    else:
        lo = torch.as_tensor(lo, device=device, dtype=dtype)

    hi = torch.as_tensor(hi, device=device, dtype=dtype)

    f_lo = fprime(lo)
    f_hi = fprime(hi)

    # Expand bracket if needed (rare but possible with extreme values)
    expand = 0
    while f_lo > 0 and expand < 50:
        # move lo downward
        hi = lo
        lo = lo - 2.0 * (torch.abs(lo) + 1.0)
        if gamma_min is not None:
            lo = torch.maximum(lo, torch.as_tensor(gamma_min, device=device, dtype=dtype))
        f_lo = fprime(lo)
        expand += 1

    expand = 0
    while f_hi < 0 and expand < 50:
        # move hi upward
        lo = hi
        hi = hi + 2.0 * (torch.abs(hi) + 1.0)
        f_hi = fprime(hi)
        expand += 1

    # Bisection
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        f_mid = fprime(mid)

        # Stop if derivative near zero or interval small
        if torch.abs(f_mid) < tol or torch.abs(hi - lo) < tol:
            gamma = mid
            break

        if f_mid > 0:
            hi = mid
        else:
            lo = mid
    else:
        gamma = 0.5 * (lo + hi)

    # Apply optional gamma_min
    if gamma_min is not None:
        gamma = torch.maximum(gamma, torch.as_tensor(gamma_min, device=device, dtype=dtype))

    # g = clip(q, [gamma, a*gamma]) elementwise
    g = torch.clamp(q, min=gamma.item(), max=(a * gamma).item())

    return g, gamma

def gmin_update_quasi(sG_hat: torch.Tensor,
                      sE_hat: torch.Tensor,
                      t: float,
                      rhoG: float,
                      rhoE: float,
                      Gmin_low: float = 1e-2,
                      Gmin_high: float = 10.0,
                      tol: float = 1e-3,
                      linearity_pcnt: Optional[float] = None) -> tuple[torch.Tensor, float]:
    """
    Update rule for Gmin in quasi-convex minimization approach.
    
    Solves:
    min_Gmin rhog/2 * sum_m max(0, Gmin - sG_hat_m)^2 
           + rhoE/2 * sum_k max(0, ||sE_hat_k||_2 - t * Gmin)^2
           
    we will solve the above with a seprate bisection search over Gmin, 
    not to be confused with the bisection search over t in the main loop.
           
    Args
    ----
    sG_hat : torch.Tensor
        tensor with shape (M,), slack consistency term for G
    sE_hat : torch.Tensor
        tensor with shape (K, D), slack consistency term for E
    t : float
        quasi-convex parameter
    rhoG : float
        penalty for G
    rhoE : float
        penalty for E
    Gmin_low : float, optional
        lower bound for Gmin
    Gmin_high : float, optional
        upper bound for Gmin
    tol : float, optional
        tolerance for the bisection search
        
    Returns
    -------
    Gmin : float
        Optimal Gmin
    """
    
    # Function
    def f(Gmin: float) -> float:
        if linearity_pcnt is None:
            diff_g = Gmin - sG_hat
            diff_e = torch.norm(sE_hat, dim=-1) - t * Gmin
            mask_g = 1.0 * (diff_g > 0)
            mask_e = 1.0 * (diff_e > 0)
            return 0.5 * rhoG * (diff_g * mask_g).square().sum(dim=-1) \
                 + 0.5 * rhoE * (diff_e * mask_e).square().sum(dim=-1) - Gmin[:, 0] # maximize Gmin
        else:
            Gmax = (1 + linearity_pcnt) * Gmin
            diff_gmin = Gmin - sG_hat
            diff_gmax = sG_hat - Gmax
            diff_e = torch.norm(sE_hat, dim=-1) - t * Gmin
            mask_gmin = 1.0 * (diff_gmin > 0)
            mask_gmax = 1.0 * (diff_gmax > 0)
            mask_e = 1.0 * (diff_e > 0)
            return 0.5 * rhoG * (diff_gmin * mask_gmin).square().sum(dim=-1) \
                 + 0.5 * rhoG * (diff_gmax * mask_gmax).square().sum(dim=-1) \
                 + 0.5 * rhoE * (diff_e * mask_e).square().sum(dim=-1) - Gmin[:, 0] # maximize Gmin
    
    # function to evaluate derivative of objective function with respect to Gmin
    def fprime(Gmin: float) -> float:
        if linearity_pcnt is None:
            diff_g = Gmin - sG_hat
            diff_e = torch.norm(sE_hat, dim=-1) - t * Gmin
            mask_g = 1.0 * (diff_g > 0)
            mask_e = 1.0 * (diff_e > 0)
            return rhoG * (diff_g * mask_g).sum(dim=-1) - t * rhoE * (diff_e * mask_e).sum(dim=-1) - 1
        else:
            Gmax = (1 + linearity_pcnt) * Gmin
            diff_gmin = Gmin - sG_hat
            diff_gmax = sG_hat - Gmax
            diff_e = torch.norm(sE_hat, dim=-1) - t * Gmin
            mask_gmin = 1.0 * (diff_gmin > 0)
            mask_gmax = 1.0 * (diff_gmax > 0)
            mask_e = 1.0 * (diff_e > 0)
            return rhoG * (diff_gmin * mask_gmin).sum(dim=-1) \
                 - rhoG * (diff_gmax * mask_gmax).sum(dim=-1) * (1 + linearity_pcnt) \
                 - t * rhoE * (diff_e * mask_e).sum(dim=-1) - 1
            
    gmins = torch.linspace(Gmin_low, Gmin_high, 500, device=sG_hat.device)
    fs = f(gmins[:, None])
    amin = fs.argmin()
    # print(f'Gmin: {gmins[amin]:1.3e}')
    return gmins[amin]
    fps = fprime(gmins[:, None])
    import matplotlib.pyplot as plt
    plt.plot(gmins.cpu(), fs.cpu())
    plt.show()
    quit()
    
    # Bisection search
    Gl = Gmin_low
    Gh = Gmin_high
    while Gh - Gl > tol:
        Gm = (Gl + Gh) / 2
        if fprime(Gm) > 0:
            Gh = Gm
        else:
            Gl = Gm
            
    return Gm
        
def admm_general(G: torch.Tensor,
                 L: torch.Tensor,
                 E: torch.Tensor,
                 P: Optional[torch.Tensor] = None,
                 C: Optional[torch.Tensor] = None,
                 d: Optional[torch.Tensor] = None,
                 F: Optional[torch.Tensor] = None,
                 g: Optional[torch.Tensor] = None,
                 t: Optional[float] = None,
                 lamdaG: Optional[float] = None,
                 Gmin: Optional[float] = None,
                 lamdaL: Optional[float] = None,
                 Lmax: Optional[float] = None,
                 lamdaE: Optional[float] = None,
                 Emax: Optional[float] = None,
                 Pmax: Optional[float] = None,
                 linearity_pcnt: Optional[float] = None,
                 state_dict: Optional[dict] = None,
                 rho: float = 1e-2,
                 admm_iters: int = 200,
                 rho_adapt: bool = False,
                 log_data: bool = True,
                 verbose: bool = True) -> dict:
    """
    Solves:
    min_(x, Gmin, Lmax, Emax) -lamda_G * Gmin + lamda_L * Lmax + lamda_E * Emax
    s.t. Gmin <= Gx
         ||Lx||_2^2 <= Lmax
         ||E_k x||_2 <= Emax for k=1...K
         ||P_w x||_2 <= Pmax for w=1...W
         |Cx| <= d
         Fx = g
         
    if linearity_pcnt is given, then:
    Gmin <= Gx <= (1 + linearity_pcnt) * Gmin.
    
    if t is given then forces:
    Emax <= t * Gmin
         
    Args
    ----
    G : torch.Tensor
        tensor with shape (M, N)
    L : torch.Tensor
        tensor with shape (N, N)
    E : torch.Tensor
        tensor with shape (K, D, N)
    P : torch.Tensor, optional
        tensor with shape (W, D, N). If None, a dummy (1, 1, N) zero matrix is used (constraint inactive).
    C : torch.Tensor, optional
        tensor with shape (Mc, N)
    d : torch.Tensor, optional
        tensor with shape (Mc,)
    F : torch.Tensor, optional
        tensor with shape (Mf, N)
    g : torch.Tensor, optional
        tensor with shape (Mf,)
    lamdaG : float, optional
        Regularization parameter for the G constraint.
    Gmin : float, optional
        Minimum value for G.
    lamdaL : float, optional
        Regularization parameter for the L constraint.
    Lmax : float, optional
        Maximum value for L.
    lamdaE : float, optional
        Regularization parameter for the E constraint.
    Emax : float, optional
        Maximum value for E.
    Pmax : float, optional
        Maximum value for ||P_w x||_2 (all w). Either lamdaP or Pmax must be provided when P is not None.
    linearity_pcnt : float, optional
        Percentage of linearity for the G constraint.
    state_dict : dict, optional
        Dictionary containing initial values for:
        - 'sG': torch.Tensor of shape (N,), slack for G
        - 'dG': torch.Tensor of shape (N,), dual  for G
        - 'sL': torch.Tensor of shape (N,), slack for L
        - 'dL': torch.Tensor of shape (N,), dual  for L
        - 'sE': torch.Tensor of shape (K, D), slack for E
        - 'dE': torch.Tensor of shape (K, D), dual  for E
        - 'sP': torch.Tensor of shape (W, D), slack for P
        - 'dP': torch.Tensor of shape (W, D), dual  for P
        - 'rhoG': float, penalty for G
        - 'rhoL': float, penalty for L
        - 'rhoE': float, penalty for E
        - 'rhoP': float, penalty for P
    rho : float, optional
        Initial ADMM penalty parameter.
    admm_iters : int, optional
        Number of ADMM iterations.
    rho_adapt : bool, optional
        Whether to use rho adaptation.
    log_data : bool, optional
        Whether to log residuals and objective values.
    verbose : bool, optional
        Whether to display progress bars.
        
    Returns
    -------
    dict
        A dictionary containing:
        - 'x': The optimized variable x with shape (N,).
        - 'r_pri': List of primal residual norms over iterations.
        - 's_dual': List of dual residual norms over iterations.
        - 'loss': List of objective values (t) over iterations.
        + all updated state_dict variables
    """
    # Consts
    K, D, N = E.shape
    M = G.shape[0]
    assert L.shape[0] == L.shape[1] == N
    torch_dev = G.device
    
    # Make sure either a lamda or a variable is provided, but not both or neither
    assert (lamdaG is not None and Gmin is None) or (lamdaG is None and Gmin is not None), \
    "Either a lamda or a variable must be provided for G."
    assert (lamdaL is not None and Lmax is None) or (lamdaL is None and Lmax is not None), \
    "Either a lamda or a variable must be provided for L."
    assert (lamdaE is not None and Emax is None) or (lamdaE is None and Emax is not None), \
    "Either a lamda or a variable must be provided for E."
    
    # Default P when not provided (inactive constraint)
    if P is None:
        P = torch.zeros((1, 3, N), device=G.device)
        Pmax = 1.0
    W, Dp, _ = P.shape
    
    # Default C, d, F, g if not provided
    if C is None or d is None:
        C = torch.zeros((1, N), device=torch_dev)
        d = torch.ones((1,), device=torch_dev)
    Mc = C.shape[0]
    if F is None or g is None:
        F = torch.zeros((1, N), device=torch_dev)
        g = torch.zeros((1,), device=torch_dev)
    Mf = F.shape[0]
    
    # G Update rule depending on if a lamda or a variable is provided
    if lamdaG is None:
        if linearity_pcnt is None:
            update_G = lambda q, rhoG: (torch.clamp(q, min=Gmin), Gmin)
        else:
            update_G = lambda q, rhoG: (torch.clamp(q, min=Gmin, max=(1 + linearity_pcnt) * Gmin), Gmin)
    else:
        if linearity_pcnt is None:
            update_G = lambda q, rhoG: g_block_update_exact(q, lamdaG, rhoG)
        else:
            update_G = lambda q, rhoG: g_block_update_band_exact(q, lamdaG, rhoG, linearity_pcnt)
            
    # L Update rule depending on if a lamda or a variable is provided
    if lamdaL is None:
         def update_L(qL, rhoL):
            Lmax_new = Lmax
            slack_new = proj_l2_ball(qL[None,], Lmax_new ** 0.5)[0]
            return slack_new, Lmax_new
    else:
        def update_L(qL, rhoL):
            slack_new = qL * rhoL / (rhoL + 2 * lamdaL)
            Lmax_new = slack_new.norm() ** 2
            return slack_new, Lmax_new
        
    # E Update rule depending on if a lamda or a variable is provided
    if lamdaE is None:
        def update_E(qE, rhoE):
            Emax_new = Emax
            slack_new = proj_l2_ball(qE, Emax_new)
            # slack_new = e_block_update_exact(qE, 0, 0, e_fixed=Emax_new)[0]
            return slack_new, Emax_new
    else:
        def update_E(qE, rhoE):
            # slack_new, Emax_new = solve_epigraph_group_l2(qE, rhoE/2/lamdaE, max_iter=20)
            slack_new, Emax_new = e_block_update_exact(qE, lamdaE, rhoE)
            return slack_new, Emax_new
        
    # Special G-E update rule for quasi-convex minimization
    if t is not None:
        def update_G_E(qG, qE, rhoG, rhoE):
            # Find optimal Gmin
            Gmin_opt = gmin_update_quasi(qG, qE, t, rhoG, rhoE, linearity_pcnt=linearity_pcnt)
            
            # Update variables
            Gmin_new = Gmin_opt
            Emax_new = t * Gmin_opt
            sG_new = torch.clamp(qG, min=Gmin_new)
            sE_new = proj_l2_ball(qE, Emax_new)
            return Gmin_new, Emax_new, sG_new, sE_new
            
    # P Update rule (same structure as E: group L2 ball)
    def update_P(qP, rhoP):
        Pmax_new = Pmax
        # slack_new = e_block_update_exact(qP, 0, 0, e_fixed=Pmax_new)[0]
        slack_new = proj_l2_ball(qP, Pmax_new)
        return slack_new, Pmax_new
    
    # Initialize variables if they are not provided in state_dict
    if state_dict is None:
        state_dict = {}
    sE = state_dict.get('sE', torch.zeros((K, D), device=torch_dev))
    dE = state_dict.get('dE', torch.zeros((K, D), device=torch_dev))
    sP = state_dict.get('sP', torch.zeros((W, Dp), device=torch_dev))
    dP = state_dict.get('dP', torch.zeros((W, Dp), device=torch_dev))
    sG = state_dict.get('sG', torch.zeros((M,), device=torch_dev))
    dG = state_dict.get('dG', torch.zeros((M,), device=torch_dev))
    sL = state_dict.get('sL', torch.zeros((N,), device=torch_dev))
    dL = state_dict.get('dL', torch.zeros((N,), device=torch_dev))
    sC = state_dict.get('sC', torch.zeros((Mc,), device=torch_dev))
    dC = state_dict.get('dC', torch.zeros((Mc,), device=torch_dev))
    sF = state_dict.get('sF', torch.zeros((Mf,), device=torch_dev))
    dF = state_dict.get('dF', torch.zeros((Mf,), device=torch_dev))
    rhoE = state_dict.get('rhoE', rho)
    rhoP = state_dict.get('rhoP', rho)
    rhoG = state_dict.get('rhoG', rho)
    rhoL = state_dict.get('rhoL', rho)
    rhoC = state_dict.get('rhoC', rho)
    rhoF = state_dict.get('rhoF', rho)
    Estack = rearrange(E, 'K D N -> (K D) N')  # (K*D, N)
    Pstack = rearrange(P, 'W D N -> (W D) N')  # (W*D, N)
    stability_I = lamda_stability * torch.eye(N, device=torch_dev)  # for numerical stability in x-update
    
    # Track diagnostics
    dct = state_dict
    if 'r_pri' not in dct and log_data:
        dct['r_pri'] = []
    if 's_dual' not in dct and log_data:
        dct['s_dual'] = []
    if 'Gmin' not in dct and log_data:
        dct['Gmin'] = []
    if 'Lmax' not in dct and log_data:
        dct['Lmax'] = []
    if 'Emax' not in dct and log_data:
        dct['Emax'] = []
        
    # ADMM iterations
    miniters = admm_iters//20
    pbar = tqdm(range(admm_iters), desc='ADMM iterations', disable=not verbose, miniters=miniters)
    for i in pbar:
        
        # x-update
        big_A = rhoE * (Estack.T @ Estack) + \
                rhoP * (Pstack.T @ Pstack) + \
                rhoG * (G.T @ G) + \
                rhoL * (L.T @ L) + \
                rhoC * (C.T @ C) + \
                rhoF * (F.T @ F) + \
                stability_I
        big_B = rhoE * Estack.T @ rearrange(sE - dE, 'K D -> (K D)') + \
                rhoP * Pstack.T @ rearrange(sP - dP, 'W D -> (W D)') + \
                rhoG * (G.T @ (sG - dG)) + \
                rhoL * (L.T @ (sL - dL)) + \
                rhoC * (C.T @ (sC - dC)) + \
                rhoF * (F.T @ (sF - dF))
        x_new = torch.linalg.solve(big_A, big_B)
        
        # G-E updates jointly
        if t is not None:
            qG = G @ x_new + dG
            qE = E @ x_new + dE
            Gmin_new, Emax_new, sG_new, sE_new = update_G_E(qG, qE, rhoG, rhoE)
        else:
            # G updates
            qG = G @ x_new + dG
            sG_new, Gmin_new = update_G(qG, rhoG)
            
            # E  updates
            qE = E @ x_new + dE
            sE_new, Emax_new = update_E(qE, rhoE)
        
        # L slack updates
        qL = L @ x_new + dL
        sL_new, Lmax_new = update_L(qL, rhoL)
        
        # P slack updates
        qP = P @ x_new + dP
        sP_new, _ = update_P(qP, rhoP)
        
        # C slack updates
        qC = C @ x_new + dC
        sC_new = torch.clamp(qC, min=-d, max=d)
        
        # F slack updates
        sF_new = g
        
        # Primal Residual
        rpG = G @ x_new - sG_new
        rpL = L @ x_new - sL_new
        rpE = E @ x_new - sE_new
        rpP = P @ x_new - sP_new
        rpC = C @ x_new - sC_new
        rpF = F @ x_new - sF_new
        
        # Dual Residual
        if log_data:
            rdG = rhoG * G.T @ (sG_new - sG)
            rdL = rhoL * L.T @ (sL_new - sL)
            rdE = rhoE * Estack.T @ (rearrange(sE_new - sE, 'K D -> (K D)'))
            rdP = rhoP * Pstack.T @ (rearrange(sP_new - sP, 'W D -> (W D)'))
            rdC = rhoC * C.T @ (sC_new - sC)
            rdF = rhoF * F.T @ (sF_new - sF)
        
        # Dual updates
        dG = dG + rpG
        dL = dL + rpL
        dE = dE + rpE
        dP = dP + rpP
        dC = dC + rpC
        dF = dF + rpF
        
        # Update variables
        x = x_new
        Gmin = Gmin_new
        Lmax = Lmax_new
        Emax = Emax_new
        sG = sG_new
        sL = sL_new
        sE = sE_new
        sP = sP_new
        sC = sC_new
        sF = sF_new
        
        # Rho adapt
        # if i % 1 == 0 and rho_adapt:
        if rho_adapt:
            rhoG, dG = boyd_update(rpG.norm(), rdG.norm(), rhoG, dG)
            rhoL, dL = boyd_update(rpL.norm(), rdL.norm(), rhoL, dL)
            rhoE, dE = boyd_update(rpE.norm(), rdE.norm(), rhoE, dE)
            rhoP, dP = boyd_update(rpP.norm(), rdP.norm(), rhoP, dP)
            rhoC, dC = boyd_update(rpC.norm(), rdC.norm(), rhoC, dC)
            rhoF, dF = boyd_update(rpF.norm(), rdF.norm(), rhoF, dF)
        
        # Diagnostics
        if log_data:
            r_norm = torch.sqrt(rpG.norm()**2 + rpL.norm()**2 + rpE.norm()**2 + rpP.norm()**2 + rpC.norm()**2).item()
            s_norm = torch.sqrt(rdG.norm()**2 + rdL.norm()**2 + rdE.norm()**2 + rdP.norm()**2 + rdC.norm()**2).item()
            dct['r_pri'].append(r_norm)
            dct['s_dual'].append(s_norm)
            dct['Gmin'].append(Gmin_new)
            dct['Lmax'].append(Lmax_new)
            dct['Emax'].append(Emax_new)
            if verbose and i % miniters == 0:
                pbar.set_description(f"Iter {i}, ||r||={r_norm:.4e}, ||s||={s_norm:.4e}")
    
    # dct['loss_pri'] = r_norm
    # dct['loss_dual'] = s_norm
    dct['x'] = x
    dct['Gmin'].append(Gmin)
    dct['Lmax'].append(Lmax)
    dct['Emax'].append(Emax)
    dct['sG'] = sG
    dct['dG'] = dG
    dct['rhoG'] = rhoG
    dct['sL'] = sL
    dct['dL'] = dL
    dct['rhoL'] = rhoL
    dct['sE'] = sE
    dct['dE'] = dE
    dct['rhoE'] = rhoE
    dct['sP'] = sP
    dct['dP'] = dP
    dct['rhoP'] = rhoP
    dct['sC'] = sC
    dct['dC'] = dC
    dct['rhoC'] = rhoC
    dct['sF'] = sF
    dct['dF'] = dF
    dct['rhoF'] = rhoF
    return dct

def admm_general_cvxpy(G: torch.Tensor,
                       L: torch.Tensor,
                       E: torch.Tensor,
                       P: Optional[torch.Tensor] = None,
                       C: Optional[torch.Tensor] = None,
                       d: Optional[torch.Tensor] = None,
                       F: Optional[torch.Tensor] = None,
                       g: Optional[torch.Tensor] = None,
                       lamdaG: Optional[float] = None,
                       Gmin: Optional[float] = None,
                       lamdaL: Optional[float] = None,
                       Lmax: Optional[float] = None,
                       lamdaE: Optional[float] = None,
                       Emax: Optional[float] = None,
                       Pmax: Optional[float] = None,
                       linearity_pcnt: Optional[float] = None,
                       state_dict: Optional[dict] = None,
                       admm_iters: int = 5_000,
                       verbose: bool = True) -> dict:
    f"""
    Solves the following:
    min_x f(x) / g(x)
    s.t. 
        x \in C
    where C [
        ||Lx||_2^2 <= Lmax
        ||P_w x||_2 <= Pmax for w=1...W
        |Cx| <= d
        Fx = g]
    f(x) = max_k ||E_k x||_2
    g(x) = min_r (G_r x)
    
    This is equivalent to the following feasibility 
    problem with bisection search on the scalar t:
    min_(x, Gmin, Emax) 0
    s.t. 
        ||E_k x||_2 <= Emax
        G_r x >= Gmin
        Emax <= t * Gmin
        Gmin >= eps
        x \in C
        
    Args:
    ----
    G : torch.Tensor
        tensor with shape (M, N)
    E : torch.Tensor
        tensor with shape (K, D, N)
    L : torch.Tensor, optional
        tensor with shape (N, N)
    P : torch.Tensor, optional
        tensor with shape (W, D, N). If None, a dummy (1, 1, N) zero matrix is used (constraint inactive).
    C : torch.Tensor, optional
        tensor with shape (Mc, N)
    d : torch.Tensor, optional
        tensor with shape (Mc,)
    F : torch.Tensor, optional  
        tensor with shape (Mf, N)
    g : torch.Tensor, optional
        tensor with shape (Mf,)
    Lmax : float, optional
        Maximum value for L.
    Pmax : float, optional
        Maximum value for ||P_w x||_2 (all w). Either lamdaP or Pmax must be provided when P is not None.
    linearity_pcnt : float, optional
        Percentage of linearity for the G constraint.
    state_dict : dict, optional
        Dictionary containing initial values for:
        - 'sG': torch.Tensor of shape (N,), slack for G
        - 'dG': torch.Tensor of shape (N,), dual  for G
        - 'sL': torch.Tensor of shape (N,), slack for L
        - 'dL': torch.Tensor of shape (N,), dual  for L
        - 'sE': torch.Tensor of shape (K, D), slack for E
        - 'dE': torch.Tensor of shape (K, D), dual  for E
        - 'sP': torch.Tensor of shape (W, D), slack for P
        - 'dP': torch.Tensor of shape (W, D), dual  for P
        - 'rhoG': float, penalty for G
        - 'rhoL': float, penalty for L
        - 'rhoE': float, penalty for E
        - 'rhoP': float, penalty for P
    admm_iters : int, optional
        Number of ADMM iterations.
    log_data : bool, optional
        Whether to log residuals and objective values.
    verbose : bool, optional
        Whether to display progress bars.
        
    Returns
    -------
    dict
    """
    # move to CPU
    torch_dev = G.device
    G = G.cpu()
    E = E.cpu()
    L = L.cpu() if L is not None else None
    P = P.cpu() if P is not None else None
    C = C.cpu() if C is not None else None
    d = d.cpu() if d is not None else None
    F = F.cpu() if F is not None else None
    g = g.cpu() if g is not None else None
    
    # Setup cvxpy 
    x = cp.Variable(G.shape[1])
    Gmin = cp.Variable(1) if Gmin is None else Gmin
    Emax = cp.Variable(1) if Emax is None else Emax
    Lmax = cp.Variable(1) if Lmax is None else Lmax
    
    # Field constraints
    field_constraints = [cp.norm(E[k] @ x, axis=0) <= Emax for k in range(len(E))]
    field_constraints += [G @ x >= Gmin]
    
    # Hardware constraints
    hw_constraints = []
    if P is not None:
        hw_constraints += [cp.norm(P[w] @ x, axis=0) <= Pmax for w in range(len(P))]
    if C is not None:
        hw_constraints += [C @ x <= d]
    if F is not None:
        hw_constraints += [F @ x == g]
    if L is not None:
        hw_constraints += [cp.square(cp.norm(L @ x)) <= Lmax]
    constraints = field_constraints + hw_constraints
    
    # Objective
    obj = 0
    if lamdaG is not None:
        obj += -lamdaG * Gmin
    if lamdaL is not None:
        obj += lamdaL * Lmax
    if lamdaE is not None:
        obj += lamdaE * Emax
    obj = cp.Minimize(obj)
    
    # Solve
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK,
               mosek_params={
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-6,
                "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-6,
                "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-6,
                "MSK_IPAR_INTPNT_MAX_ITERATIONS": 50,
                "MSK_IPAR_NUM_THREADS": 8,  # adjust to your machine
                },
               verbose=verbose)
    
def quasi_convex_min_ratio(G: torch.Tensor,
                           E: torch.Tensor,
                           tstart: float = 10.0,
                           L: Optional[torch.Tensor] = None,
                           P: Optional[torch.Tensor] = None,
                           C: Optional[torch.Tensor] = None,
                           d: Optional[torch.Tensor] = None,
                           F: Optional[torch.Tensor] = None,
                           g: Optional[torch.Tensor] = None,
                           Lmax: Optional[float] = None,
                           Pmax: Optional[float] = None,
                           linearity_pcnt: Optional[float] = None,
                           state_dict: Optional[dict] = None,
                           rho: float = 1e-2,
                           bisection_iters: int = 7,
                           admm_iters: int = 5_000,
                           admm_iters_reduced: int = 5_000,
                           rho_adapt: bool = False,
                           log_data: bool = True,
                           verbose: bool = True) -> dict:
    f"""
    Solves the following:
    min_x f(x) / g(x)
    s.t. 
        x \in C
    where C [
        ||Lx||_2^2 <= Lmax
        ||P_w x||_2 <= Pmax for w=1...W
        |Cx| <= d
        Fx = g]
    f(x) = max_k ||E_k x||_2
    g(x) = min_r (G_r x)
    
    This is equivalent to the following feasibility 
    problem with bisection search on the scalar t:
    min_(x, Gmin, Emax) 0
    s.t. 
        ||E_k x||_2 <= Emax
        G_r x >= Gmin
        Emax <= t * Gmin
        Gmin >= eps
        x \in C
        
    Args:
    ----
    G : torch.Tensor
        tensor with shape (M, N)
    L : torch.Tensor
        tensor with shape (N, N)
    E : torch.Tensor
        tensor with shape (K, D, N)
    P : torch.Tensor, optional
        tensor with shape (W, D, N). If None, a dummy (1, 1, N) zero matrix is used (constraint inactive).
    C : torch.Tensor, optional
        tensor with shape (Mc, N)
    d : torch.Tensor, optional
        tensor with shape (Mc,)
    F : torch.Tensor, optional
        tensor with shape (Mf, N)
    g : torch.Tensor, optional
        tensor with shape (Mf,)
    Lmax : float, optional
        Maximum value for L.
    Pmax : float, optional
        Maximum value for ||P_w x||_2 (all w). Either lamdaP or Pmax must be provided when P is not None.
    linearity_pcnt : float, optional
        Percentage of linearity for the G constraint.
    state_dict : dict, optional
        Dictionary containing initial values for:
        - 'sG': torch.Tensor of shape (N,), slack for G
        - 'dG': torch.Tensor of shape (N,), dual  for G
        - 'sL': torch.Tensor of shape (N,), slack for L
        - 'dL': torch.Tensor of shape (N,), dual  for L
        - 'sE': torch.Tensor of shape (K, D), slack for E
        - 'dE': torch.Tensor of shape (K, D), dual  for E
        - 'sP': torch.Tensor of shape (W, D), slack for P
        - 'dP': torch.Tensor of shape (W, D), dual  for P
        - 'rhoG': float, penalty for G
        - 'rhoL': float, penalty for L
        - 'rhoE': float, penalty for E
        - 'rhoP': float, penalty for P
    rho : float, optional
        Initial ADMM penalty parameter.
    bisection_iters : int, optional
        Number of bisection iterations.
    admm_iters : int, optional
        Number of ADMM iterations.
    rho_adapt : bool, optional
        Whether to use rho adaptation.
    log_data : bool, optional
        Whether to log residuals and objective values.
    verbose : bool, optional
        Whether to display progress bars.
        
    Returns
    -------
    dict
    """
    # # To CPU for cvxpy
    # torch_dev = G.device
    # G = G.cpu()
    # E = E.cpu()
    # L = L.cpu() if L is not None else None
    # P = P.cpu() if P is not None else None
    # C = C.cpu() if C is not None else None
    # d = d.cpu() if d is not None else None
    # F = F.cpu() if F is not None else None
    # g = g.cpu() if g is not None else None
    # dct =  quasi_convex_min_ratio_cvxpy(G=G, E=E, tstart=tstart, 
    #                                     L=L, P=P, C=C, d=d, F=F, g=g, 
    #                                     Lmax=Lmax, Pmax=Pmax, linearity_pcnt=linearity_pcnt, 
    #                                     state_dict=state_dict, rho=rho, t_tol=t_tol, 
    #                                     admm_iters=admm_iters, admm_iters_reduced=admm_iters_reduced, 
    #                                     rho_adapt=rho_adapt, log_data=log_data, verbose=verbose)
    # dct['x'] = dct['x'].to(torch_dev)
    # return dct

    # Get shapes from G
    M, N = G.shape
    K, D, _ = E.shape
    
    dct_ret = None
    def is_feasible(x: torch.Tensor, 
                    Emax: float, 
                    Gmin: float, 
                    pcnt_tol: float = 0.05,
                    abs_tol: float = 1e-3,
                    verbose: bool = True) -> bool:
        
        if linearity_pcnt is None:
            # Gradient should be smaller than Gmin
            Gmin_actual = (G @ x).min()
            if Gmin_actual < Gmin * (1 - pcnt_tol):
                if verbose:
                    print(f"Gx({Gmin_actual:.2f}) < Gmin({Gmin * (1 - pcnt_tol):.2f})")
                return False
        else:
            Gmin_actual = (G @ x).min()
            Gmax_actual = (G @ x).max()
            Gmax = Gmin * (1 + linearity_pcnt)
            if Gmin_actual < Gmin * (1 - pcnt_tol):
                if verbose:
                    print(f"Gx({Gmin_actual:.2f}) < Gmin({Gmin * (1 - pcnt_tol):.2f})")
                return False
            if Gmax_actual > Gmax * (1 + pcnt_tol):
                if verbose:
                    print(f"Gx({Gmax_actual:.2f}) > Gmax({Gmax * (1 + pcnt_tol):.2f})")
                return False
        
        # Peak Efield should be smaller than Emax
        Emax_actual = (E @ x).norm(dim=-1).max()
        if Emax_actual > Emax * (1 + pcnt_tol):
            if verbose:
                print(f"||E_k x||_2({Emax_actual:.2f}) > Emax({Emax * (1 + pcnt_tol):.2f})")
            return False
        
        # L should be smaller than Lmax
        if L is not None:
            Lmax_actual = (L @ x).norm().square()
            if Lmax_actual > Lmax * (1 + pcnt_tol):
                if verbose:
                    print(f"||Lx||_2^2({Lmax_actual:.3e}) > Lmax({Lmax * (1 + pcnt_tol):.2f})")
                return False
            
        # P should be smaller than Pmax
        if P is not None:
            Pmax_actual = (P @ x).norm(dim=-1).max()
            if Pmax_actual > Pmax * (1 + pcnt_tol):
                if verbose:
                    print(f"||P_w x||_2({Pmax_actual:.3e}) > Pmax({Pmax * (1 + pcnt_tol):.2f})")
                return False
            
        # C should be smaller than d
        if C is not None:
            Cx_actual = (C @ x).abs()
            if (Cx_actual - d * (1 + pcnt_tol)).max() > 0:
                if verbose:
                    idx_violation = torch.argwhere((Cx_actual - d * (1 + pcnt_tol)) > 0)[:, 0][0]
                    print(f"|Cx|({Cx_actual[idx_violation]:.3e}) > d({d[idx_violation] * (1 + pcnt_tol):.2f}")
                return False
            
        # F should be equal to g
        if F is not None:
            Fx_actual = (F @ x)
            Fxg = Fx_actual - g
            if Fxg.abs().max() > abs_tol:
                if verbose:
                    idx_violation = torch.argwhere(Fxg.abs() > abs_tol)[:, 0][0]
                    sign = torch.sign(Fx_actual[idx_violation] - g[idx_violation])
                    if sign > 0:
                        print(f"Fx == g constriant violated:\nFx({Fx_actual[idx_violation]:.3e}) > g({abs_tol + g[idx_violation]:.3e})")
                    else:
                        print(f"Fx == g constriant violated:\nFx({Fx_actual[idx_violation]:.3e}) < g({-abs_tol + g[idx_violation]:.3e})")
                return False
            
        print(f'Gmin={Gmin_actual:.2f}, Emax={Emax_actual:.2f}, constraints satisfied.')
            
        return True
        
    # Starting bounds
    th = 2 * tstart
    tl = 0
    assert (th + tl) / 2 == tstart, "Starting bounds must be symmetric around tstart"
    
    # Main loop
    dct = state_dict
    pbar = tqdm(range(bisection_iters), desc='Bisection iterations', disable=not verbose)
    for k in pbar:
        
        # Bisection search
        t = (th + tl) / 2
        
        # Solve the feasibility problem
        dct = admm_general(G=G, L=L, E=E, C=C, d=d, F=F, g=g, P=P,
                           lamdaG=0, lamdaE=0, lamdaL=None,
                           Lmax=Lmax, Pmax=Pmax, 
                           linearity_pcnt=linearity_pcnt, 
                           t=t,
                        #    state_dict=dct, 
                           rho=rho, 
                           admm_iters=admm_iters if k == 0 else admm_iters_reduced, 
                           rho_adapt=rho_adapt if k == 0 else False, 
                           log_data=log_data, 
                           verbose=verbose)# if k == 0 else False)
        
        # Check feasibility of solution 
        # if k == bisection_iters - 2:
        #     breakpoint()
        if is_feasible(dct['x'], dct['Emax'][-1], dct['Gmin'][-1]):
            feas = True
            th = t
            dct_ret = dct
        else:
            feas = False
            tl = t
        t_diff = th - tl
        
        # Update progress bar
        pbar.set_description(f"Bisection Iter {k}, t={t:1.3f}, feas={feas}")
        
    return dct_ret

def quasi_convex_min_ratio_cvxpy(G: torch.Tensor,
                                 E: torch.Tensor,
                                 tstart: float = 10.0,
                                 L: Optional[torch.Tensor] = None,
                                 P: Optional[torch.Tensor] = None,
                                 C: Optional[torch.Tensor] = None,
                                 d: Optional[torch.Tensor] = None,
                                 F: Optional[torch.Tensor] = None,
                                 g: Optional[torch.Tensor] = None,
                                 Lmax: Optional[float] = None,
                                 Pmax: Optional[float] = None,
                                 linearity_pcnt: Optional[float] = None,
                                 state_dict: Optional[dict] = None,
                                 rho: float = 1e-2,
                                 t_tol: float = 1e-3,
                                 admm_iters: int = 5_000,
                                 admm_iters_reduced: int = 5_000,
                                 rho_adapt: bool = False,
                                 log_data: bool = True,
                                 verbose: bool = True) -> dict:
    f"""
    Solves the following:
    min_x f(x) / g(x)
    s.t. 
        x \in C
    where C [
        ||Lx||_2^2 <= Lmax
        ||P_w x||_2 <= Pmax for w=1...W
        |Cx| <= d
        Fx = g]
    f(x) = max_k ||E_k x||_2
    g(x) = min_r (G_r x)
    
    This is equivalent to the following feasibility 
    problem with bisection search on the scalar t:
    min_(x, Gmin, Emax) 0
    s.t. 
        ||E_k x||_2 <= Emax
        G_r x >= Gmin
        Emax <= t * Gmin
        Gmin >= eps
        x \in C
        
    Args:
    ----
    G : torch.Tensor
        tensor with shape (M, N)
    L : torch.Tensor
        tensor with shape (N, N)
    E : torch.Tensor
        tensor with shape (K, D, N)
    P : torch.Tensor, optional
        tensor with shape (W, D, N). If None, a dummy (1, 1, N) zero matrix is used (constraint inactive).
    C : torch.Tensor, optional
        tensor with shape (Mc, N)
    d : torch.Tensor, optional
        tensor with shape (Mc,)
    F : torch.Tensor, optional
        tensor with shape (Mf, N)
    g : torch.Tensor, optional
        tensor with shape (Mf,)
    Lmax : float, optional
        Maximum value for L.
    Pmax : float, optional
        Maximum value for ||P_w x||_2 (all w). Either lamdaP or Pmax must be provided when P is not None.
    linearity_pcnt : float, optional
        Percentage of linearity for the G constraint.
    state_dict : dict, optional
        Dictionary containing initial values for:
        - 'sG': torch.Tensor of shape (N,), slack for G
        - 'dG': torch.Tensor of shape (N,), dual  for G
        - 'sL': torch.Tensor of shape (N,), slack for L
        - 'dL': torch.Tensor of shape (N,), dual  for L
        - 'sE': torch.Tensor of shape (K, D), slack for E
        - 'dE': torch.Tensor of shape (K, D), dual  for E
        - 'sP': torch.Tensor of shape (W, D), slack for P
        - 'dP': torch.Tensor of shape (W, D), dual  for P
        - 'rhoG': float, penalty for G
        - 'rhoL': float, penalty for L
        - 'rhoE': float, penalty for E
        - 'rhoP': float, penalty for P
    rho : float, optional
        Initial ADMM penalty parameter.
    admm_iters : int, optional
        Number of ADMM iterations.
    rho_adapt : bool, optional
        Whether to use rho adaptation.
    log_data : bool, optional
        Whether to log residuals and objective values.
    verbose : bool, optional
        Whether to display progress bars.
        
    Returns
    -------
    dict
    """
    # Get shapes from G
    M, N = G.shape
    K, D, _ = E.shape
    
    def is_feasible(x: torch.Tensor, 
                    Emax: float, 
                    Gmin: float, 
                    pcnt_tol: float = 0.05) -> bool:
        
        # Gradient should be smaller than Gmin
        Gmin_actual = (G @ x).min()
        if Gmin_actual < Gmin * (1 - pcnt_tol):
            return False
        
        # Peak Efield should be smaller than Emax
        Emax_actual = (E @ x).norm(dim=-1).max()
        if Emax_actual > Emax * (1 + pcnt_tol):
            return False
        
        # L should be smaller than Lmax
        if L is not None:
            Lmax_actual = (L @ x).norm().square()
            if Lmax_actual > Lmax * (1 + pcnt_tol):
                return False
            
        # P should be smaller than Pmax
        if P is not None:
            Pmax_actual = (P @ x).norm(dim=-1).max()
            if Pmax_actual > Pmax * (1 + pcnt_tol):
                return False
            
        # C should be smaller than d
        if C is not None:
            Cx_actual = (C @ x).abs()
            if (Cx_actual - d * (1 + pcnt_tol)).max() > 0:
                return False
            
        # F should be equal to g
        if F is not None:
            Fxg_actual = (F @ x - g).abs()
            if (Fxg_actual - g * pcnt_tol).max() > 0:
                return False
            
        return True
        
    # Starting bounds
    th = 2 * tstart
    tl = 0
    assert (th + tl) / 2 == tstart, "Starting bounds must be symmetric around tstart"
    t_diff = float('inf')
    
    # Main loop
    k = 0
    dct = state_dict
    while t_diff > t_tol:
        
        # Bisection search
        t = (th + tl) / 2
        
        # Setup cvxpy 
        x = cp.Variable(N)
        Gmin = cp.Variable(1)
        Emax = cp.Variable(1)
        
        # Field constraints
        field_constraints = [cp.norm(E[k] @ x, axis=0) <= Emax for k in range(len(E))]
        field_constraints += [G @ x >= Gmin]
        field_constraints += [Emax <= t * Gmin]
        field_constraints += [Gmin >= 0.1]
        
        # Hardware constraints
        hw_constraints = []
        if P is not None:
            hw_constraints += [cp.norm(P[w] @ x, axis=0) <= Pmax for w in range(len(P))]
        if C is not None:
            hw_constraints += [C @ x <= d]
        if F is not None:
            hw_constraints += [F @ x == g]
        if L is not None:
            hw_constraints += [cp.square(cp.norm(L @ x)) <= Lmax]
        constraints = field_constraints + hw_constraints
        
        # Solve the feasibility problem
        # prob = cp.Problem(cp.Maximize(0), constraints)
        prob = cp.Problem(cp.Maximize(Gmin), constraints)
        failed = False
        try:
            prob.solve(
                solver=cp.MOSEK,
                mosek_params={
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-6,
                    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-6,
                    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-6,
                    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 50,
                    "MSK_IPAR_NUM_THREADS": 8,  # adjust to your machine
                },
                verbose=False,
            )
        except:
            failed = True
        
        if prob.status == 'infeasible' or failed:
            feas = False
        else:
            # double check feasibility
            x_val = torch.from_numpy(x.value).type(torch.float32)
            Gmin_val = torch.from_numpy(Gmin.value).type(torch.float32)
            Emax_val = torch.from_numpy(Emax.value).type(torch.float32)
            feas = is_feasible(x_val, Emax_val, Gmin_val)
            
        # Update bounds
        if feas:
            th = t
        else:
            tl = t
        
        # Update iteration counter
        k += 1
        t_diff = th - tl
        if verbose:
            print(f"Bisection Iter {k}, t={t:1.3f}, t_diff={t_diff:.3e}, feas={feas}")
    
    dct = {
        'x': x_val,
        'Gmin': [Gmin_val],
        'Emax': [Emax_val],
        'Lmax': [Lmax],
        'r_pri': [],
        's_dual': [],
    }
    return dct

def unrolled_admm_general(thetas: list[nn.Parameter],
                          G_theta: Callable[[nn.Parameter], torch.Tensor],
                          L_theta: Callable[[nn.Parameter], torch.Tensor],
                          E_theta: Callable[[nn.Parameter], torch.Tensor],
                          P_theta: Callable[[nn.Parameter], torch.Tensor] = lambda x : None,
                          C_theta: Callable[[nn.Parameter], torch.Tensor] = lambda x : None,
                          F_theta: Callable[[nn.Parameter], torch.Tensor] = lambda x : None,
                          loss_theta: Optional[Callable[[nn.Parameter], float]] = None,
                          d: Optional[torch.Tensor] = None,
                          g: Optional[torch.Tensor] = None,
                          lamdaG: Optional[float] = None,
                          Gmin: Optional[float] = None,
                          lamdaL: Optional[float] = None,
                          Lmax: Optional[float] = None,
                          lamdaE: Optional[float] = None,
                          Emax: Optional[float] = None,
                          Pmax: Optional[float] = None,
                          linearity_pcnt: Optional[float] = None,
                          state_dict: Optional[dict] = None,
                          rho: float = 1e-2,
                          rho_adapt: bool = False,
                          lr: Union[float, list[float]] = 1e-3,
                          epochs: int = 100,
                          admm_iters: int = 50,
                          log_data: bool = True,
                          verbose: bool = True) -> dict:
    """
    Unrolled ADMM optimization for a general problem.
    
    Args
    ----
    thetas : list[nn.Parameter]
        List of parameters to optimize.
    G_theta : Callable[[nn.Parameter], torch.Tensor]
        Function to compute the G constraint.
    L_theta : Callable[[nn.Parameter], torch.Tensor]
        Function to compute the L constraint.
    E_theta : Callable[[nn.Parameter], torch.Tensor]
        Function to compute the E constraint.
    P_theta : Callable[[nn.Parameter], torch.Tensor]
        Function to compute the P constraint.
    C_theta : Callable[[nn.Parameter], torch.Tensor]
        Function to compute the C constraint.
    F_theta : Callable[[nn.Parameter], torch.Tensor]
        Function to compute the F constraint.
    g : Optional[torch.Tensor]
        Optional fixed g value.
    loss_theta : Optional[Callable[[nn.Parameter], float]]
        Function to compute the loss.
    d : Optional[torch.Tensor]
        Optional fixed d value.
    lamdaG : Optional[float]
        Optional fixed lamdaG value.
    Gmin : Optional[float]
        Optional fixed Gmin value.
    lamdaL : Optional[float]
        Optional fixed lamdaL value.
    Lmax : Optional[float]
        Optional fixed Lmax value.
    lamdaE : Optional[float]
        Optional fixed lamdaE value.
    Emax : Optional[float]
        Optional fixed Emax value.
    Pmax : Optional[float]
        Optional fixed Pmax value.
    linearity_pcnt : Optional[float]
        Optional fixed linearity_pcnt value.
    state_dict : Optional[dict]
        Optional initial state dictionary.
    rho : float
        Initial ADMM penalty parameter.
    rho_adapt : bool
        Whether to use rho adaptation.
    lr : float
        Learning rate.
    epochs : int
        Number of epochs.
    admm_iters : int
        Number of ADMM iterations.
    log_data : bool
        Whether to log residuals and objective values.
    verbose : bool
        Whether to display progress bars.
        
    Returns
    -------
    dict
        A dictionary containing:
        - 'x': The optimized variable x with shape (N,).
        - 'r_pri': List of primal residual norms over iterations.
        - 's_dual': List of dual residual norms over iterations.
        - 'loss': List of objective values over iterations.
        + all updated state_dict variables
    """
    # Get shape from G
    Gtemp = G_theta(thetas)
    torch_dev = Gtemp.device
    _, N = Gtemp.shape
    
    # Optimizer
    if isinstance(lr, list):
        opt = torch.optim.Adam([{'params': [thetas[i]], 'lr': lr[i]} for i in range(len(thetas))])
    else:
        opt = torch.optim.Adam(thetas, lr=lr)
    
    # Initial state
    if state_dict is None:
        state_dict = {'coeffs': []}
    else:
        state_dict['coeffs'] = [state_dict['x'].detach().cpu()]
    
    # Initialize theta lists
    for i in range(len(thetas)):
        if 'theta_'+str(i) not in state_dict:
            state_dict['theta_'+str(i)] = []
    
    # Default loss function
    if loss_theta is None:
        loss_theta = lambda x : 0.0
        
    # Default C, d
    if d is None:
        d = torch.ones((1,), device=torch_dev)
        C_theta = lambda thetas : torch.zeros((1, N), device=torch_dev)
    
    # Main optimizer loop
    for e in tqdm(range(epochs), desc='Adam Epochs'):
        # Zero gradients
        opt.zero_grad()
        
        # Generate matrices
        G = G_theta(thetas)
        L = L_theta(thetas)
        E = E_theta(thetas)
        C = C_theta(thetas)
        P = P_theta(thetas)
        F = F_theta(thetas)
                
        # ADMM solve
        state_dict = admm_general(G=G, L=L, E=E, C=C, d=d, F=F, g=g, P=P,
                                  lamdaG=lamdaG, Gmin=Gmin, lamdaL=lamdaL, Lmax=Lmax, lamdaE=lamdaE, Emax=Emax, Pmax=Pmax, linearity_pcnt=linearity_pcnt, state_dict=state_dict, rho=rho, admm_iters=admm_iters, rho_adapt=rho_adapt, log_data=log_data, verbose=verbose)
        
        # Minimize gradient loss function 
        loss = 0
        if lamdaG is not None:
            # Gminval = state_dict['Gmin'][-1]
            Gminval = (G @ state_dict['x']).min()
            loss -= lamdaG * Gminval
        if lamdaL is not None:
            # Lmaxval = state_dict['Lmax'][-1]
            Lmaxval = (L @ state_dict['x']).norm()**2
            loss += lamdaL * Lmaxval
        if lamdaE is not None:
            # Emaxval = state_dict['Emax'][-1]
            Emaxval = (E @ state_dict['x']).norm(dim=-1).max()
            loss += lamdaE * Emaxval
        # loss += 1e5 * (state_dict['r_pri'][-1] + state_dict['s_dual'][-1])
        
            
        # Loss on theta parameters
        loss += loss_theta(thetas)

        # Backpropagation
        loss.backward()
        for _ in range(1):
            opt.step()
            
        # Track thetas and coeffs
        if log_data and e % 10 == 0:    
            state_dict['coeffs'].append(state_dict['x'].detach().cpu())
            for i in range(len(thetas)):
                state_dict['theta_'+str(i)].append(thetas[i].detach().cpu())
                
        # Detach all admm variables
        for k in state_dict.keys():
            if isinstance(state_dict[k], torch.Tensor):
                state_dict[k] = state_dict[k].detach()
            

    return state_dict
