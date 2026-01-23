from __future__ import annotations
from typing import Literal, Optional
import torch
from torch import nn

BoundaryMode = Literal["clamp", "wrap", "zero"]

class BSpline1D(nn.Module):
    """
    Uniform cubic B-spline model: y(x) = sum_j c_j * B3((x - x0)/h - j)

    - Coefficients are learnable (nn.Parameter) and directly accessible at .coeff
    - Fast, vectorized evaluation for arbitrary-shaped x tensors
    - Simple .fit(...) to match (x_i, y_i)
    """

    def __init__(
        self,
        n_coeff: int,
        xmin: float = 0.0,
        xmax: float = 1.0,
        boundary: BoundaryMode = "clamp",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            n_coeff: number of control values c_j (j = 0..n_coeff-1)
            x0: leftmost grid origin for knots/cells
            h: uniform spacing between adjacent basis centers
            boundary: 'clamp' (repeat ends), 'wrap' (periodic), or 'zero' (outside -> 0)
        """
        super().__init__()
        assert n_coeff >= 4, "Need at least 4 coefficients for cubic B-splines."
        self.n = n_coeff
        self.x0 = xmin
        self.h = (xmax - xmin) / (n_coeff - 1)
        self.boundary: BoundaryMode = boundary
        
        self.coeff = nn.Parameter(torch.zeros(n_coeff, device=device, dtype=dtype), 
                                  requires_grad=False)

    # ---------- Public API ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the spline at arbitrary-shaped x (broadcast-safe).
        Returns a tensor shaped like x."""
        idx, w, mask = self._indices_and_weights(x)
        # gather 4 neighboring coefficients and weight-sum
        # idx, w, mask are (..., 4)
        c = self._gather_coeff(idx)  # (..., 4)
        if mask is not None:         # "zero" mode: null out out-of-range contributions
            w = w * mask
        y = (w * c).sum(dim=-1)
        return y

    def dy(self, x: torch.Tensor) -> torch.Tensor:
        """First derivative f'(x). Same broadcasting rules as forward()."""
        # Recompute t, i, u to get analytic derivative weights
        x = x.to(self.coeff.device, self.coeff.dtype)
        t = (x - self.x0) / self.h
        i = torch.floor(t).to(torch.long)
        u = (t - i).clamp(0, 1 - torch.finfo(x.dtype).eps)

        u2 = u * u

        # derivative weights dB/du for neighbors [i-1, i, i+1, i+2]
        dw0 = -0.5 * (1 - u) * (1 - u)
        dw1 = 1.5 * u2 - 2.0 * u
        dw2 = -1.5 * u2 + 1.0 * u + 0.5
        dw3 = 0.5 * u2

        idx = torch.stack([i - 1, i, i + 1, i + 2], dim=-1)
        dw  = torch.stack([dw0, dw1, dw2, dw3], dim=-1)

        if self.boundary == "wrap":
            idx = idx % self.n
            mask = None
        elif self.boundary == "clamp":
            idx = idx.clamp(0, self.n - 1)
            mask = None
        elif self.boundary == "zero":
            off_left  = idx < 0
            off_right = idx >= self.n
            mask = ~(off_left | off_right)
            idx = idx.clamp(0, self.n - 1)
            dw  = dw * mask.to(dw.dtype)
        else:
            raise ValueError(f"Unknown boundary mode: {self.boundary}")

        c = self._gather_coeff(idx)  # (..., 4)
        return (dw * c).sum(dim=-1) / self.h

    def d2y(self, x: torch.Tensor) -> torch.Tensor:
        """Second derivative f''(x)."""
        x = x.to(self.coeff.device, self.coeff.dtype)
        t = (x - self.x0) / self.h
        i = torch.floor(t).to(torch.long)
        u = (t - i).clamp(0, 1 - torch.finfo(x.dtype).eps)

        # second derivatives d2B/du2
        d2w0 = (1 - u)
        d2w1 = 3.0 * u - 2.0
        d2w2 = -3.0 * u + 1.0
        d2w3 = u

        idx = torch.stack([i - 1, i, i + 1, i + 2], dim=-1)
        d2w = torch.stack([d2w0, d2w1, d2w2, d2w3], dim=-1)

        if self.boundary == "wrap":
            idx = idx % self.n
            mask = None
        elif self.boundary == "clamp":
            idx = idx.clamp(0, self.n - 1)
            mask = None
        elif self.boundary == "zero":
            off_left  = idx < 0
            off_right = idx >= self.n
            mask = ~(off_left | off_right)
            idx = idx.clamp(0, self.n - 1)
            d2w = d2w * mask.to(d2w.dtype)
        else:
            raise ValueError(f"Unknown boundary mode: {self.boundary}")

        c = self._gather_coeff(idx)
        return (d2w * c).sum(dim=-1) / (self.h * self.h)

    @torch.no_grad()
    def fit_lstsq(self, x: torch.Tensor, y: torch.Tensor, lam: float = 0.0) -> None:
        """
        Fit coefficients to data by (optionally-regularized) least squares:
            minimize ||Phi c - y||^2 + lam * ||D c||^2
        where Phi is the B-spline design matrix and D is a 2nd-difference matrix.

        This builds a dense Phi (N x n) — great for N up to a few 1e4.
        For very large N, use `fit_optim` instead.
        """
        Phi = self.design_matrix_dense(x)  # [N, n]
        N, n = Phi.shape
        yv = y.reshape(N, 1)

        if lam > 0:
            # second-difference smoothing regularizer
            D = torch.zeros((self.n - 2, self.n), device=Phi.device, dtype=Phi.dtype)
            ar = torch.arange(self.n - 2, device=Phi.device)
            D[ar, ar]     = 1.0
            D[ar, ar + 1] = -2.0
            D[ar, ar + 2] = 1.0
            # normal equations: (Phi^T Phi + lam D^T D) c = Phi^T y
            A = Phi.T @ Phi + lam * (D.T @ D)
            b = Phi.T @ yv
            c, _ = torch.linalg.solve(A, b), True
        else:
            # pesudo-inverse solution: c = (Phi^T Phi)^(-1) Phi^T y
            c = torch.linalg.pinv(Phi) @ yv   # [n,1]

        self.coeff.copy_(c.squeeze(-1))

    @torch.no_grad()
    def fit_optim(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lam: float = 0.0,
        lr: float = 0.1,
        steps: int = 200,
        optimizer_cls=torch.optim.LBFGS,
    ) -> None:
        """
        Optimizer-based fit (handles huge N; no big dense matrix).
        Adds optional 2nd-diff smoothing with weight `lam`.
        """
        opt = optimizer_cls([self.coeff], lr=lr)

        def reg2(c):
            return ((c[:-2] - 2 * c[1:-1] + c[2:]) ** 2).mean() if c.numel() > 2 else c.sum() * 0.0

        def closure():
            opt.zero_grad()
            yhat = self.forward(x)
            loss = torch.mean((yhat - y) ** 2)
            if lam > 0:
                loss = loss + lam * reg2(self.coeff)
            loss.backward()
            return loss

        for _ in range(steps):
            opt.step(closure)

    @torch.no_grad()
    def design_matrix_dense(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dense (N x n) design matrix Phi s.t. Phi @ coeff ≈ y(x).
        Efficiently built via 4-way scatter; suitable for moderate N.
        """
        x_flat = x.reshape(-1)
        idx, w, mask = self._indices_and_weights(x_flat)   # [N,4] each
        N = x_flat.numel()
        Phi = torch.zeros((N, self.n), device=x.device, dtype=x.dtype)

        for k in range(4):
            col = idx[:, k]
            wk  = w[:, k]
            if mask is not None:
                wk = wk * mask[:, k]
            # scatter-add into Phi rows
            Phi[torch.arange(N, device=x.device), col] += wk

        return Phi

    # ---------- Internals ----------
    def _gather_coeff(self, idx: torch.Tensor) -> torch.Tensor:
        # idx shape: (..., 4); returns coeff gathered per last dim
        # Use take_along_dim by expanding coeff
        c = self.coeff
        # expand coeff to match leading shape (broadcast)
        flat_idx = idx.reshape(-1, 4)
        gathered = c.index_select(0, flat_idx.reshape(-1)).reshape(flat_idx.shape)
        return gathered.reshape(idx.shape)

    def _indices_and_weights(self, x: torch.Tensor):
        """
        For each x, compute (i-1, i, i+1, i+2) and their cubic B-spline weights.

        Let u = (x - (x0 + i*h)) / h in [0,1).
        Weights for neighbors [i-1, i, i+1, i+2]:
            w0 = (1 - u)^3 / 6
            w1 = (3u^3 - 6u^2 + 4) / 6
            w2 = (-3u^3 + 3u^2 + 3u + 1) / 6
            w3 = u^3 / 6
        """
        x = x.to(self.coeff.device, self.coeff.dtype)
        # base cell index
        t = (x - self.x0) / self.h
        i = torch.floor(t).to(torch.long)  # ... base index
        u = (t - i).clamp(0, 1 - torch.finfo(x.dtype).eps)  # numerical safety, in [0,1)

        u2 = u * u
        u3 = u2 * u

        w0 = ((1 - u) ** 3) / 6.0
        w1 = (3 * u3 - 6 * u2 + 4) / 6.0
        w2 = (-3 * u3 + 3 * u2 + 3 * u + 1) / 6.0
        w3 = (u3) / 6.0

        base = i
        idx = torch.stack([base - 1, base, base + 1, base + 2], dim=-1)  # (..., 4)
        w   = torch.stack([w0, w1, w2, w3], dim=-1)                      # (..., 4)

        if self.boundary == "wrap":
            idx = idx % self.n
            mask = None
        elif self.boundary == "clamp":
            idx = idx.clamp(0, self.n - 1)
            mask = None
        elif self.boundary == "zero":
            # keep original idx for mask, clamp for gather, then zero out off-range weights
            off_left  = idx < 0
            off_right = idx >= self.n
            mask = ~(off_left | off_right)
            idx = idx.clamp(0, self.n - 1)
            mask = mask.to(w.dtype)
        else:
            raise ValueError(f"Unknown boundary mode: {self.boundary}")

        return idx, w, mask


# torch_dev = torch.device('cpu')

# # 1) Make a spline over [x0, x0 + (n-1)h]
# spl = BSpline1D(n_coeff=15, xmin=-1, xmax=1, boundary="clamp", dtype=torch.float32, device=torch_dev)

# # 2) Fit to data (x,y) — closed-form least squares (fast for moderate N)
# x = torch.linspace(-1, 1, steps=15, device=torch_dev)
# y = torch.sin(2 * torch.pi * x)
# spl.fit_lstsq(x, y, lam=1e-2)   # lam adds smoothness via 2nd-diff penalty

# # 3) Query anywhere, with any shape
# xq = x
# yq = spl(xq)   # shape (4, 3, 256), runs fully on GPU

# # 4) Access or modify coefficients directly
# coeff = spl.coeff         # nn.Parameter
# coeff.data *= 0.9         # (example tweak)

# import matplotlib
# matplotlib.use('webAgg')
# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(coeff.data.cpu())

# plt.figure(figsize=(14,7))
# plt.plot(x.cpu(), y.cpu(), 'k.', alpha=0.1, label='data')
# xx = torch.linspace(-1, 1, steps=500, device=torch_dev)
# yy = spl(xx)
# dy = spl.dy(xx)
# plt.plot(xx.cpu(), yy.cpu(), 'r-', label='spline fit')
# plt.plot(xx.cpu(), dy.cpu(), 'g-', label='spline derivative')
# plt.title('BSpline1D fit example')
# plt.legend()
# plt.show()