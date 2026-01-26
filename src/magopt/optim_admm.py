import torch

from torch import nn
from tqdm import tqdm
from typing import Optional, Callable
from einops import rearrange

# Constants
lamda_stability = 1e-12  # for numerical stability in x-update

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

@torch.no_grad()
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

@torch.no_grad()
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

@torch.no_grad()
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

def admm_general(G: torch.Tensor,
                 L: torch.Tensor,
                 E: torch.Tensor,
                 C: Optional[torch.Tensor] = None,
                 d: Optional[torch.Tensor] = None,
                 lamdaG: Optional[float] = None,
                 Gmin: Optional[float] = None,
                 lamdaL: Optional[float] = None,
                 Lmax: Optional[float] = None,
                 lamdaE: Optional[float] = None,
                 Emax: Optional[float] = None,
                 linearity_pcnt: Optional[float] = None,
                 state_dict: Optional[dict] = None,
                 rho: float = 1e-2,
                 admm_iters: int = 200,
                 rho_adapt: bool = False,
                 log_data: bool = True,
                 verbose: bool = True) -> dict:
    f"""
    Solves:
    min_(x, Gmin, Lmax, Emax) lamda_G * Gmin + lamda_L * Lmax + lamda_E * Emax
    s.t. Gmin <= Gx
         ||Lx||_2^2 <= Lmax
         ||E_k x||_2 <= Emax for k=1...K
         |Cx| <= d
         
    if linearity_pcnt is given, then:
    Gmin <= Gx <= (1 + linearity_pcnt) * Gmin.
         
    Args
    ----
    G : torch.Tensor
        tensor with shape (M, N)
    L : torch.Tensor
        tensor with shape (N, N)
    E : torch.Tensor
        tensor with shape (K, D, N)
    C : torch.Tensor, optional
        tensor with shape (Mc, N)
    d : torch.Tensor, optional
        tensor with shape (Mc,)
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
    linearity_pcnt : float, optional
        Percentage of linearity for the G constraint.
    state_dict : dict, optional
        Dictionary containing initial values for:
        - 'sG': torch.Tensor of shape (N,), slack for G
        - 'dG': torch.Tensor of shape (N,), dual  for G
        - 'sL': torch.Tensor of shape (N,), slack for L
        - 'dL': torch.Tensor of shape (N,), dual  for L
        - 'sE': torch.Tensor of shape (K,), slack for E
        - 'dE': torch.Tensor of shape (K,), dual  for E
        - 'rhoG': float, penalty for G
        - 'rhoL': float, penalty for L
        - 'rhoE': float, penalty for E
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
    
    # Default C, d if not provided
    if C is None:
        C = torch.zeros((1, N), device=torch_dev)
    if d is None:
        d = torch.ones((1,), device=torch_dev)
    Mc = C.shape[0]
    
    # G Update rule depending on if a lamda or a variable is provided
    if lamdaG is None:
        if linearity_pcnt is None:
            update_G = lambda q, rhoG: torch.clamp(q, min=Gmin)
        else:
            update_G = lambda q, rhoG: torch.clamp(q, min=Gmin, max=(1 + linearity_pcnt) * Gmin)
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
            return slack_new, Emax_new
    else:
        def update_E(qE, rhoE):
            slack_new, Emax_new = solve_epigraph_group_l2(qE, rhoE/2/lamdaE, max_iter=20)
            return slack_new, Emax_new
    
    # Initialize variables if they are not provided in state_dict
    if state_dict is None:
        state_dict = {}
    sE = state_dict.get('sE', torch.zeros((K, D), device=torch_dev))
    dE = state_dict.get('dE', torch.zeros((K, D), device=torch_dev))
    sG = state_dict.get('sG', torch.zeros((M,), device=torch_dev))
    dG = state_dict.get('dG', torch.zeros((M,), device=torch_dev))
    sL = state_dict.get('sL', torch.zeros((N,), device=torch_dev))
    dL = state_dict.get('dL', torch.zeros((N,), device=torch_dev))
    sC = state_dict.get('sC', torch.zeros((Mc,), device=torch_dev))
    dC = state_dict.get('dC', torch.zeros((Mc,), device=torch_dev))
    rhoE = state_dict.get('rhoE', rho)
    rhoG = state_dict.get('rhoG', rho)
    rhoL = state_dict.get('rhoL', rho)
    rhoC = state_dict.get('rhoC', rho)
    Estack = rearrange(E, 'K D N -> (K D) N')  # (K*D, N)
    stability_I = lamda_stability * torch.eye(N, device=torch_dev)  # for numerical stability in x-update
    
    # Track diagnostics
    dct = state_dict
    if 'r_pri' not in dct and log_data:
        dct['r_pri'] = []
    if 's_dual' not in dct and log_data:
        dct['s_dual'] = []
    if 'Em' not in dct and log_data:
        dct['Gmin'] = []
    if 'Lmax' not in dct and log_data:
        dct['Lmax'] = []
    if 'Emax' not in dct and log_data:
        dct['Emax'] = []
        
    # ADMM iterations
    for i in tqdm(range(admm_iters), desc='ADMM iterations', disable=not verbose):
        
        # x-update
        big_A = rhoE * (Estack.T @ Estack) + \
                rhoG * (G.T @ G) + \
                rhoL * (L.T @ L) + \
                rhoC * (C.T @ C) + \
                stability_I
        big_B = rhoE * Estack.T @ rearrange(sE - dE, 'K D -> (K D)') + \
                rhoG * (G.T @ (sG - dG)) + \
                rhoL * (L.T @ (sL - dL)) + \
                rhoC * (C.T @ (sC - dC))
        x_new = torch.linalg.solve(big_A, big_B)
        
        # G slack updates
        qG = G @ x_new + dG
        sG_new, Gmin_new = update_G(qG, rhoG)
        
        # L slack updates
        qL = L @ x_new + dL
        sL_new, Lmax_new = update_L(qL, rhoL)
        
        # E slack updates
        qE = E @ x_new + dE
        sE_new, Emax_new = update_E(qE, rhoE)
        
        # C slack updates
        qC = C @ x_new + dC
        sC_new = torch.clamp(qC, min=-d, max=d)
        
        # Primal Residual
        rpG = G @ x_new - sG_new
        rpL = L @ x_new - sL_new
        rpE = E @ x_new - sE_new
        rpC = C @ x_new - sC_new
        
        # Dual Residual
        if log_data:
            rdG = rhoG * G.T @ (sG_new - sG)
            rdL = rhoL * L.T @ (sL_new - sL)
            rdE = rhoE * Estack.T @ (rearrange(sE_new - sE, 'K D -> (K D)'))
            rdC = rhoC * C.T @ (sC_new - sC)
        
        # Dual updates
        dG = dG + rpG
        dL = dL + rpL
        dE = dE + rpE
        dC = dC + rpC
        
        # Update variables
        x = x_new
        Gmin = Gmin_new
        Lmax = Lmax_new
        Emax = Emax_new
        sG = sG_new
        sL = sL_new
        sE = sE_new
        sC = sC_new
        
        # Rho adapt
        if i % 10 == 0 and rho_adapt:
            rhoG, dG = boyd_update(rpG.norm(), rdG.norm(), rhoG, dG)
            rhoL, dL = boyd_update(rpL.norm(), rdL.norm(), rhoL, dL)
            rhoE, dE = boyd_update(rpE.norm(), rdE.norm(), rhoE, dE)
            rhoC, dC = boyd_update(rpC.norm(), rdC.norm(), rhoC, dC)
        
        # Diagnostics
        if log_data:
            r_norm = torch.sqrt(rpG.norm()**2 + rpL.norm()**2 + rpE.norm()**2 + rpC.norm()**2).item()
            s_norm = torch.sqrt(rdG.norm()**2 + rdL.norm()**2 + rdE.norm()**2 + rdC.norm()**2).item()
            dct['r_pri'].append(r_norm)
            dct['s_dual'].append(s_norm)
            dct['Gmin'].append(Gmin_new)
            dct['Lmax'].append(Lmax_new)
            dct['Emax'].append(Emax_new)
            if verbose and i % 50 == 0:
                print(f"Iter {i}, ||r||={r_norm:.4e}, ||s||={s_norm:.4e}")
            
    dct['x'] = x
    dct['sG'] = sG
    dct['dG'] = dG
    dct['rhoG'] = rhoG
    dct['sL'] = sL
    dct['dL'] = dL
    dct['rhoL'] = rhoL
    dct['sE'] = sE
    dct['dE'] = dE
    dct['rhoE'] = rhoE
    dct['sC'] = sC
    dct['dC'] = dC
    dct['rhoC'] = rhoC
    return dct
    
def admm_min_quadratic_peak_norm_constraint(L: torch.Tensor,
                                            A: torch.Tensor,
                                            T: torch.Tensor,
                                            b_lower: torch.Tensor,
                                            b_upper: torch.Tensor,
                                            t: float,
                                            state_dict: Optional[dict] = None,
                                            rho: float = 1e-2,
                                            admm_iters: int = 200,
                                            rho_adapt: bool = False,
                                            log_data: bool = True,
                                            verbose: bool = True) -> dict:
    """
    Solves:
    min_{x, t} x^T L x
    s.t. b_l <= Ax <= b_u
         ||T_k x||_2 <= t for k=1...K
         
    Args
    ----
    L : torch.Tensor
        tensor with shape (N, N)
    A : torch.Tensor
        tensor with shape (M, N)
    T : torch.Tensor
        tensor with shape (K, D)
    b_lower : torch.Tensor
        tensor with shape (M,)
    b_upper : torch.Tensor
        tensor with shape (M,)
    t : float
        Peak norm constraint
    state_dict : dict, optional
        Dictionary containing initial values for:
        - 'sT': torch.Tensor of shape (K, D), slack for T
        - 'dT': torch.Tensor of shape (K, D), dual  for T
        - 'sA': torch.Tensor of shape (M,), slack for A
        - 'dA': torch.Tensor of shape (M,), dual  for A
        - 'rhoT': float, penalty for T
        - 'rhoA': float, penalty for A
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
    # Get shapes
    K, D, N = T.shape
    N = L.shape[0]
    M = A.shape[0]
    torch_dev = A.device
    
    # Initialize variables if they are not provided in state_dict
    if state_dict is None:
        state_dict = {}
    sT = state_dict.get('sT', torch.zeros((K, D), device=torch_dev))
    dT = state_dict.get('dT', torch.zeros((K, D), device=torch_dev))
    sA = state_dict.get('sA', torch.zeros((M,), device=torch_dev))
    dA = state_dict.get('dA', torch.zeros((M,), device=torch_dev))
    rhoT = state_dict.get('rhoT', rho)
    rhoA = state_dict.get('rhoA', rho)
    Tstack = rearrange(T, 'K D N -> (K D) N')  # (K*D, N)
    stability_I = lamda_stability * torch.eye(N, device=torch_dev)  # for numerical stability in x-update
    
    # ADMM iterations
    dct = state_dict
    if 'r_pri' not in dct and log_data:
        dct['r_pri'] = []
    if 's_dual' not in dct and log_data:
        dct['s_dual'] = []
    if 'loss_me' not in dct and log_data:
        dct['loss_me'] = []
    for i in tqdm(range(admm_iters), desc='ADMM iterations', disable=not verbose):
        
       # x-update
        big_A = rhoT * Tstack.T @ Tstack + \
                rhoA * (A.T @ A) + \
                (L + L.T) + \
                stability_I
        big_B = rhoT * Tstack.T @ rearrange(sT - dT, 'K D -> (K D)') + \
                rhoA * (A.T @ (sA - dA))
        x_new = torch.linalg.solve(big_A, big_B)
        
        # T slack updates
        qT = T @ x_new + dT
        sT_new = proj_l2_ball(qT, t)
        
        # A slack updates
        qA = A @ x_new + dA
        sA_new = torch.clamp(qA, b_lower, b_upper)
        
        # Primal Residual
        rpT = T @ x_new - sT_new
        rpA = A @ x_new - sA_new
        
        # Dual Residual
        if log_data:
            rdT = rhoT * Tstack.T @ (rearrange(sT_new - sT, 'K D -> (K D)'))
            rdA = rhoA * A.T @ (sA_new - sA)
        
        # Dual updates
        dT = dT + rpT
        dA = dA + rpA
        
        # Update variables
        x = x_new
        sT = sT_new
        sA = sA_new
        
        # Rho adapt
        if i % 10 == 0 and rho_adapt:
            rhoT, dT = boyd_update(rpT.norm(), rdT.norm(), rhoT, dT)
            rhoA, dA = boyd_update(rpA.norm(), rdA.norm(), rhoA, dA)
        
        # Diagnostics
        if log_data:
            r_norm = torch.sqrt(rpT.norm()**2 + rpA.norm()**2).item()
            s_norm = torch.sqrt(rdT.norm()**2 + rdA.norm()**2).item()
            dct['r_pri'].append(r_norm)
            dct['s_dual'].append(s_norm)
            dct['loss_me'].append((x_new @ L @ x_new).item())
            if verbose and i % 50 == 0:
                print(f"Iter {i}: loss={dct['loss_me'][-1]:.4f}, ||r||={r_norm:.4e}, ||s||={s_norm:.4e}")

    dct['x'] = x
    dct['sA'] = sA
    dct['dA'] = dA
    dct['rhoA'] = rhoA
    return dct

def admm_min_quadratic(L: torch.Tensor,
                       A: torch.Tensor,
                       b_lower: torch.Tensor,
                       b_upper: torch.Tensor,
                       state_dict: Optional[dict] = None,
                       rho: float = 1e-2,
                       admm_iters: int = 200,
                       rho_adapt: bool = False,
                       log_data: bool = True,
                       verbose: bool = True) -> dict:
    """
    Solves:
    min_{x, t} x^T L x
    s.t. b_l <= Ax <= b_u
         
    Args
    ----
    L : torch.Tensor
        tensor with shape (N, N)
    A : torch.Tensor
        tensor with shape (M, N)
    b_lower : torch.Tensor
        tensor with shape (M,)
    b_upper : torch.Tensor
        tensor with shape (M,)
    state_dict : dict, optional
        Dictionary containing initial values for:
        - 'sA': torch.Tensor of shape (M,), slack for A
        - 'dA': torch.Tensor of shape (M,), dual  for A
        - 'rhoT': float, penalty for T
        - 'rhoA': float, penalty for A
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
    # Get shapes
    N = L.shape[0]
    M = A.shape[0]
    torch_dev = A.device
    
    # Initialize variables if they are not provided in state_dict
    if state_dict is None:
        state_dict = {}
    sA = state_dict.get('sA', torch.zeros((M,), device=torch_dev))
    dA = state_dict.get('dA', torch.zeros((M,), device=torch_dev))
    rhoA = state_dict.get('rhoA', rho)
    stability_I = lamda_stability * torch.eye(N, device=torch_dev)  # for numerical stability in x-update
    
    # ADMM iterations
    dct = state_dict
    if 'r_pri' not in dct and log_data:
        dct['r_pri'] = []
    if 's_dual' not in dct and log_data:
        dct['s_dual'] = []
    if 'loss_me' not in dct and log_data:
        dct['loss_me'] = []
    for i in tqdm(range(admm_iters), desc='ADMM iterations', disable=not verbose):
        
        # x-update
        big_A = rhoA * (A.T @ A) + \
                (L + L.T) + \
                stability_I
        big_B = rhoA * (A.T @ (sA - dA))
        x_new = torch.linalg.solve(big_A, big_B)
        
        # A slack updates
        qA = A @ x_new + dA
        sA_new = torch.clamp(qA, b_lower, b_upper)
        
        # Primal Residual
        rpA = A @ x_new - sA_new
        
        # Dual Residual
        if log_data:
            rdA = rhoA * A.T @ (sA_new - sA)
        
        # Dual updates
        dA = dA + rpA
        
        # Update variables
        x = x_new
        sA = sA_new
        
        # Rho adapt
        if i % 10 == 0 and rho_adapt:
            rhoA, dA = boyd_update(rpA.norm(), rdA.norm(), rhoA, dA)
        
        # Diagnostics
        if log_data:
            r_norm = rpA.norm().item()
            s_norm = rdA.norm().item()
            dct['r_pri'].append(r_norm)
            dct['s_dual'].append(s_norm)
            dct['loss_me'].append((x_new @ L @ x_new).item())
            if verbose and i % 50 == 0:
                print(f"Iter {i}: loss={dct['loss_me'][-1]:.4f}, ||r||={r_norm:.4e}, ||s||={s_norm:.4e}")

    dct['x'] = x
    dct['sA'] = sA
    dct['dA'] = dA
    dct['rhoA'] = rhoA
    return dct

def admm_min_peak_norm_plus_quadtratic(T: torch.Tensor,
                                       L: torch.Tensor,
                                       A: torch.Tensor,
                                       b_lower: torch.Tensor,
                                       b_upper: torch.Tensor,
                                       lamda: float,
                                       C: Optional[torch.Tensor] = None,
                                       d: Optional[torch.Tensor] = None,
                                       state_dict: Optional[dict] = None,
                                       rho: float = 1e-2,
                                       admm_iters: int = 200,
                                       rho_adapt: bool = False,
                                       log_data: bool = True,
                                       verbose: bool = True) -> dict:
    """
    Solves:
    min_{x, t} t + lamda * x^T L x
    s.t. ||T_k x||_2 <= t for k=1...K
         b_l <= Ax <= b_u
         
    Args
    ----
    T : torch.Tensor
        tensor with shape (K, D, N)
    L : torch.Tensor
        tensor with shape (N, N)
    A : torch.Tensor
        tensor with shape (M, N)
    b_lower : torch.Tensor
        tensor with shape (M,)
    b_upper : torch.Tensor
        tensor with shape (M,)
    lamda : float
        Regularization parameter for the quadratic term.
    C : torch.Tensor, optional
        tensor with shape (P, N) for additional infinity-norm constraints ||C x||_inf <= d
    d : torch.Tensor, optional
        tensor with shape (P,) for additional infinity-norm constraints ||C x||_inf <= d
    state_dict : dict, optional
        Dictionary containing initial values for:
        - 'sT': torch.Tensor of shape (K, D), slack for T
        - 'dT': torch.Tensor of shape (K, D), dual  for T
        - 'sA': torch.Tensor of shape (M,), slack for A
        - 'dA': torch.Tensor of shape (M,), dual  for A
        - 'rhoT': float, penalty for T
        - 'rhoA': float, penalty for A
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
        - 't': The optimized peak norm t (scalar).
        - 'r_pri': List of primal residual norms over iterations.
        - 's_dual': List of dual residual norms over iterations.
        - 'loss': List of objective values (t) over iterations.
        + all updated state_dict variables
    """
    # Get shapes
    K, D, N = T.shape
    M = A.shape[0]
    torch_dev = T.device
    
    # Default values 
    if C is None:
        C = torch.zeros((1, N), device=torch_dev)
        d = torch.tensor([1e12], device=torch_dev)  # effectively no constraint
    
    # Initialize variables if they are not provided in state_dict
    if state_dict is None:
        state_dict = {}
    sT = state_dict.get('sT', torch.zeros((K, D), device=torch_dev))
    dT = state_dict.get('dT', torch.zeros((K, D), device=torch_dev))
    sA = state_dict.get('sA', torch.zeros((M,), device=torch_dev))
    dA = state_dict.get('dA', torch.zeros((M,), device=torch_dev))
    sC = state_dict.get('sC', torch.zeros((C.shape[0],), device=torch_dev))
    dC = state_dict.get('dC', torch.zeros((C.shape[0],), device=torch_dev))
    rhoT = state_dict.get('rhoT', rho)
    rhoA = state_dict.get('rhoA', rho)
    rhoC = state_dict.get('rhoC', rho)
    Tstack = rearrange(T, 'K D N -> (K D) N')  # (K*D, N)
    stability_I = lamda_stability * torch.eye(N, device=torch_dev)  # for numerical stability in x-update
    
    # ADMM iterations
    dct = state_dict
    if 'r_pri' not in dct and log_data:
        dct['r_pri'] = []
    if 's_dual' not in dct and log_data:
        dct['s_dual'] = []
    if 'loss_t' not in dct and log_data:
        dct['loss_t'] = []
    if 'loss_me' not in dct and log_data:
        dct['loss_me'] = []
    for i in tqdm(range(admm_iters), desc='ADMM iterations', disable=not verbose):
        
        # x-update
        big_A = rhoT * Tstack.T @ Tstack + \
                rhoA * (A.T @ A) + \
                rhoC * (C.T @ C) + \
                lamda * (L + L.T) + \
                stability_I
        big_B = rhoT * Tstack.T @ rearrange(sT - dT, 'K D -> (K D)') + \
                rhoA * (A.T @ (sA - dA)) + \
                rhoC * (C.T @ (sC - dC))
        x_new = torch.linalg.solve(big_A, big_B)
        
        # T slack updates and t update
        qT = T @ x_new + dT
        sT_new, t_new = solve_epigraph_group_l2(qT, rhoT/2, max_iter=20)
        
        # A slack updates
        qA = A @ x_new + dA
        sA_new = torch.clamp(qA, b_lower, b_upper)  
        
        # C slack updates
        qC = C @ x_new + dC
        sC_new = torch.clamp(qC, min=-d, max=d)
        
        # Primal Residual
        rpT = T @ x_new - sT_new
        rpA = A @ x_new - sA_new
        rpC = C @ x_new - sC_new
        
        # Dual Residual
        if log_data:
            rdT = rhoT * Tstack.T @ (rearrange(sT_new - sT, 'K D -> (K D)'))
            rdA = rhoA * A.T @ (sA_new - sA)
            rdC = rhoC * C.T @ (sC_new - sC)
        
        # Dual updates
        dT = dT + rpT
        dA = dA + rpA
        dC = dC + rpC
        
        # Update variables
        x = x_new
        t = t_new
        sT = sT_new
        sA = sA_new
        sC = sC_new
        
        # Rho adapt
        if i % 10 == 0 and rho_adapt:
            rhoT, dT = boyd_update(rpT.norm(), rdT.norm(), rhoT, dT)
            rhoA, dA = boyd_update(rpA.norm(), rdA.norm(), rhoA, dA)
            rhoC, dC = boyd_update(rpC.norm(), rdC.norm(), rhoC, dC)
        
        # Diagnostics
        if log_data:
            r_norm = torch.sqrt(rpT.norm()**2 + rpA.norm()**2 + rpC.norm()**2).item()
            s_norm = torch.sqrt(rdT.norm()**2 + rdA.norm()**2 + rdC.norm()**2).item()
            dct['r_pri'].append(r_norm)
            dct['s_dual'].append(s_norm)
            dct['loss_t'].append(t.item())
            dct['loss_me'].append((x_new @ L @ x_new).item())
            if verbose and i % 50 == 0:
                print(f"Iter {i}: t={t.item():.4f}, ||r||={r_norm:.4e}, ||s||={s_norm:.4e}")
            
    dct['x'] = x
    dct['t'] = t
    dct['sT'] = sT
    dct['dT'] = dT
    dct['rhoT'] = rhoT
    dct['sA'] = sA
    dct['dA'] = dA
    dct['rhoA'] = rhoA
    dct['sC'] = sC
    dct['dC'] = dC
    dct['rhoC'] = rhoC
    return dct
    
def admm_min_peak_norm(T: torch.Tensor,
                       A: torch.Tensor,
                       b_lower: torch.Tensor,
                       b_upper: torch.Tensor,
                       C: torch.Tensor,
                       d: torch.Tensor,
                       L: torch.Tensor,
                       L_max: float,
                       state_dict: Optional[dict] = None,
                       rho: float = 1e-2,
                       admm_iters: int = 200,
                       rho_adapt: bool = False,
                       log_data: bool = True,
                       verbose: bool = True) -> dict:
    """
    Solves:
    min_{x, t} t
    s.t. ||T_k x||_2 <= t for k=1...K
         b_l <= Ax <= b_u
         ||C x_n||_inf <= d
         x^T L x <= L_max
         
    Args
    ----
    T : torch.Tensor
        tensor with shape (K, D, N)
    A : torch.Tensor
        tensor with shape (M, N)
    b_lower : torch.Tensor
        tensor with shape (M,)
    b_upper : torch.Tensor
        tensor with shape (M,)
    C : torch.Tensor
        tensor with shape (P, N)
    d : torch.Tensor
        tensor with shape (P,)
    L : torch.Tensor
        tensor with shape (N, N)
    L_max : float
        Upper bound on the quadratic form x^T L x
    state_dict : dict, optional
        Dictionary containing initial values for:
        - 'sT': torch.Tensor of shape (K, D), slack for T
        - 'dT': torch.Tensor of shape (K, D), dual  for T
        - 'sA': torch.Tensor of shape (M,), slack for A
        - 'dA': torch.Tensor of shape (M,), dual  for A
        - 'sC': torch.Tensor of shape (P,), slack for C
        - 'dC': torch.Tensor of shape (P,), dual  for C
        - 'rhoT': float, penalty for T
        - 'rhoA': float, penalty for A
        - 'rhoC': float, penalty for C
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
        - 't': The optimized peak norm t (scalar).
        - 'r_pri': List of primal residual norms over iterations.
        - 's_dual': List of dual residual norms over iterations.
        - 'loss': List of objective values (t) over iterations.
        + all new variables from state_dict
    """
    
    # Get shapes
    K, D, N = T.shape
    M = A.shape[0]
    P = C.shape[0]
    torch_dev = T.device
    
    # Initialize variables if they are not provided in state_dict
    if state_dict is None:
        state_dict = {}
    sT = state_dict.get('sT', torch.zeros((K, D), device=torch_dev))
    dT = state_dict.get('dT', torch.zeros((K, D), device=torch_dev))
    sA = state_dict.get('sA', torch.zeros((M,), device=torch_dev))
    dA = state_dict.get('dA', torch.zeros((M,), device=torch_dev))
    sC = state_dict.get('sC', torch.zeros((P,), device=torch_dev))
    dC = state_dict.get('dC', torch.zeros((P,), device=torch_dev))
    sL = state_dict.get('sL', torch.zeros((N,), device=torch_dev))
    dL = state_dict.get('dL', torch.zeros((N,), device=torch_dev))
    rhoT = state_dict.get('rhoT', rho)
    rhoA = state_dict.get('rhoA', rho)
    rhoC = state_dict.get('rhoC', rho)
    rhoL = state_dict.get('rhoL', rho)
    Tstack = rearrange(T, 'K D N -> (K D) N')  # (K*D, N)
    I = torch.eye(N, device=torch_dev)
    stability_I = lamda_stability * I # for numerical stability in x-update
    
    # ADMM iterations
    dct = state_dict
    if 'r_pri' not in dct and log_data:
        dct['r_pri'] = []
    if 's_dual' not in dct and log_data:
        dct['s_dual'] = []
    if 'loss_t' not in dct and log_data:
        dct['loss_t'] = []
    for i in tqdm(range(admm_iters), desc='ADMM iterations', disable=not verbose):
        
        # x-update
        big_A = rhoT * Tstack.T @ Tstack + \
                rhoA * (A.T @ A) + \
                rhoC * (C.T @ C) + \
                rhoL * I + \
                stability_I
        big_B = rhoT * Tstack.T @ rearrange(sT - dT, 'K D -> (K D)') + \
                rhoA * (A.T @ (sA - dA)) + \
                rhoC * (C.T @ (sC - dC)) + \
                rhoL * (sL - dL)
        x_new = torch.linalg.solve(big_A, big_B)
        
        # T slack updates and t update
        qT = T @ x_new + dT
        sT_new, t_new = solve_epigraph_group_l2(qT, rhoT/2, max_iter=20)
        
        # A slack updates
        qA = A @ x_new + dA
        sA_new = torch.clamp(qA, b_lower, b_upper)
        
        # C slack updates
        qC = C @ x_new + dC
        sC_new = torch.clamp(qC, min=-d, max=d)
        
        # L slack updates
        qL = x_new + dL
        sL_new = proj_ellipsoid(qL, L, L_max)
        
        # Primal Residual
        rpT = T @ x_new - sT_new
        rpA = A @ x_new - sA_new
        rpC = C @ x_new - sC_new
        rpL = x_new - sL_new
        
        # Dual Residual
        if log_data:
            rdT = rhoT * Tstack.T @ (rearrange(sT_new - sT, 'K D -> (K D)'))
            rdA = rhoA * A.T @ (sA_new - sA)
            rdC = rhoC * C.T @ (sC_new - sC)
            rdL = rhoL * (sL_new - sL)
        
        # Dual updates
        dT = dT + rpT
        dA = dA + rpA
        dC = dC + rpC
        dL = dL + rpL
        
        # Update variables
        x = x_new
        t = t_new
        sT = sT_new
        sA = sA_new
        sC = sC_new
        sL = sL_new
        
        # Rho adapt
        if i % 10 == 0 and rho_adapt:
            rhoT, dT = boyd_update(rpT.norm(), rdT.norm(), rhoT, dT)
            rhoA, dA = boyd_update(rpA.norm(), rdA.norm(), rhoA, dA)
            rhoC, dC = boyd_update(rpC.norm(), rdC.norm(), rhoC, dC)
            rhoL, dL = boyd_update(rpL.norm(), rdL.norm(), rhoL, dL)
            
        # Diagnostics
        if log_data:
            r_norm = torch.sqrt(rpT.norm()**2 + rpA.norm()**2 + rpC.norm()**2 + rpL.norm()**2).item()
            s_norm = torch.sqrt(rdT.norm()**2 + rdA.norm()**2 + rdC.norm()**2 + rdL.norm()**2).item()
            dct['r_pri'].append(r_norm)
            dct['s_dual'].append(s_norm)
            dct['loss_t'].append(t.item())
            if verbose and i % 50 == 0:
                print(f"Iter {i}: t={t.item():.4f}, ||r||={r_norm:.4e}, ||s||={s_norm:.4e}")
    
    dct['x'] = x
    dct['t'] = t
    dct['sT'] = sT
    dct['dT'] = dT
    dct['rhoT'] = rhoT
    dct['sA'] = sA
    dct['dA'] = dA
    dct['rhoA'] = rhoA
    dct['sC'] = sC
    dct['dC'] = dC
    dct['rhoC'] = rhoC
    dct['sL'] = sL
    dct['dL'] = dL
    dct['rhoL'] = rhoL
    return dct
    
def unrolled_admm(thetas: list[nn.Parameter],
                  T_theta: Callable[[nn.Parameter], torch.Tensor],
                  A_theta: Callable[[nn.Parameter], torch.Tensor],
                  b_lower: torch.Tensor,
                  b_upper: torch.Tensor,
                  C_theta: Optional[Callable[[nn.Parameter], torch.Tensor]] = None,
                  d: Optional[torch.Tensor] = None,
                  L_theta: Optional[Callable[[nn.Parameter], torch.Tensor]] = None,
                  loss_theta: Optional[Callable[[nn.Parameter], float]] = None,
                  t: Optional[float] = None,
                  lamda : Optional[float] = None,
                  rho_adapt: bool = False,
                  lr: float = 1e-3,
                  epochs: int = 100,
                  admm_iters: int = 50,
                  log_data: bool = True) -> dict:
    """
    Unrolls ADMM for optimizing matrix parameters theta.
    The T, A, C, and L matrices are functions of theta.
    
    Args
    ----
    thetas : list[nn.Parameter]
        List of parameters to optimize.
    T_theta : callable
        Function that takes in thetas and returns T with shape (K, D, N).
    A_theta : callable
        Function that takes in thetas and returns A with shape (M, N).
    b_lower : torch.Tensor
        Tensor with shape (M,).
    b_upper : torch.Tensor
        Tensor with shape (M,).
    C_theta : callable, optional
        Function that takes in thetas and returns C with shape (P, N).
    d : torch.Tensor, optional
        Tensor with shape (P,).
    L_theta : callable, optional
        Function that takes in thetas and returns L with shape (N, N).
    loss_theta : callable, optional
        Function that takes in thetas and returns a scalar loss.
    t : float, optional
        peak norm constraint.
    lamda : float, optional
        Regularization parameter for the quadratic term if M_theta is provided.
    rho_adapt : bool, optional
        Whether to use rho adaptation in ADMM.
    lr : float, optional
        Learning rate for the optimizer.
    epochs : int, optional
        Number of outer optimization epochs.
    admm_iters : int, optional
        Number of ADMM iterations per epoch.
    log_data : bool, optional
        Whether to log residuals and objective values.
        
    Returns
    -------
    dict
        A dictionary containing:
        - 'thetas': The optimized parameters.
        - 'x': The optimized variable x with shape (N,).
        - 't': The optimized peak norm t (scalar).
        - 'r_pri': List of primal residual norms over epochs.
        - 's_dual': List of dual residual norms over epochs.
        - 'loss': List of objective values (t) over epochs.
    """
    # Optimizer
    opt = torch.optim.Adam(thetas, lr=lr)
    
    # Initial state
    state_dict = {'coeffs': []}
    for i in range(len(thetas)):
        state_dict['theta_'+str(i)] = []
    rho_init = 1e-2
    
    # Default loss function
    if loss_theta is None:
        loss_theta = lambda x : 0.0
    
    # Main optimizer loop
    for e in tqdm(range(epochs), desc='Adam Epochs'):
        # Zero gradients
        opt.zero_grad()
        
        # Generate matrices
        T = T_theta(thetas)
        A = A_theta(thetas)
        if C_theta is not None and d is not None:
            C = C_theta(thetas)
        else:
            C = None
            d = None
        if L_theta is not None:
            L = L_theta(thetas)
        else:
            L = None
                
        # ADMM solve
        if L_theta is not None and lamda is not None:
            state_dict = admm_min_peak_norm_plus_quadtratic(T, L, A, b_lower, b_upper,
                                                            lamda=lamda,
                                                            C=C,
                                                            d=d,
                                                            rho=rho_init,
                                                            admm_iters=admm_iters,
                                                            state_dict=state_dict,
                                                            rho_adapt=rho_adapt,
                                                            log_data=log_data,
                                                            verbose=False)
        elif L_theta is not None and t is not None:
            state_dict = admm_min_quadratic_peak_norm_constraint(L, A, T, b_lower, b_upper,
                                                                 t=t,
                                                                 rho=rho_init,
                                                                 admm_iters=admm_iters,
                                                                 state_dict=state_dict,
                                                                 rho_adapt=rho_adapt,
                                                                 log_data=log_data,
                                                                 verbose=False)
        else:
            raise ValueError("Either C_theta and d or M_theta must be provided.")
        
        # Minimize gradient loss function 
        if L_theta is None:
            loss = state_dict['t'] 
        elif t is None:
            x = state_dict['x']
            loss = state_dict['t'] + lamda * (x @ L @ x)
        else:
            x = state_dict['x']
            loss = x @ L @ x
            
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
        
        