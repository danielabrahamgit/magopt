"""
Stream-function coil on a parametric surface r(u, v).

This mirrors ``stream_func_coil`` but uses a :class:`~magopt.gradient_coils.gradient_surfaces.surface`
instance for geometry. Parameters ``u`` and ``v`` are taken on a fixed interval (by default ``[0, 1]``
for both), matching the convention in ``gradient_surfaces`` (e.g. ``elliptical_frustum``).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from einops import einsum
from matplotlib import cm
from skimage import measure
from scipy.interpolate import interp1d
from tqdm import tqdm

from .gradient_coil import gradient_coil
from .gradient_surfaces import surface
from .stream_func_coil import (
    _chebyshev_bases,
    _deriv_chebyshev_bases,
    _deriv_fourier_bases,
    _fourier_bases,
)


class stream_func_surface_coil(gradient_coil):
    """
    Current sheet on a surface parameterized by (u, v), with stream function
    expanded in tensor products of 1D bases in u and v 

    """

    def __init__(
        self,
        grad_surface: surface,
        u_min: float = 0.0,
        u_max: float = 1.0,
        v_min: float = 0.0,
        v_max: float = 1.0,
        num_v_modes: int = 10,
        num_u_modes: int = 15,
        num_u: int = 200,
        num_v: int = 200,
        u_bases: str = "fourier",
        v_bases: str = "chebyshev",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32):
        self.grad_surface = grad_surface

        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        u0 = torch.tensor(self.u_min, dtype=dtype, device=device)
        v0 = torch.tensor(self.v_min, dtype=dtype, device=device)
        p = grad_surface.to_xyz(u0, v0)
        self.torch_dev = p.device
        self.dtype = p.dtype

        self.num_u = int(num_u)
        self.num_v = int(num_v)

        self.M = int(num_u_modes)
        self.K = int(num_v_modes)

        self._ub = u_bases
        self._vb = v_bases

        # v bases (like z in stream_func_coil): index K
        if v_bases == "chebyshev":
            ks = torch.arange(self.K, device=self.torch_dev)
            self.v_bases = lambda x: _chebyshev_bases(x, self.K, ks)
            self.v_bases_deriv = lambda x: _deriv_chebyshev_bases(x, self.K, ks)
        elif v_bases == "fourier":
            ks = torch.arange(int(math.ceil(self.K / 2)), device=self.torch_dev)
            self.v_bases = lambda x: _fourier_bases(x * torch.pi, self.K, ks)
            self.v_bases_deriv = lambda x: _deriv_fourier_bases(x * torch.pi, self.K, ks) * torch.pi
        else:
            raise ValueError(f"Invalid v_bases: {v_bases}")

        # u bases (like theta in stream_func_coil): index M
        if u_bases == "fourier":
            ms = torch.arange(int(math.ceil(self.M / 2)), device=self.torch_dev)
            self.u_bases = lambda x: _fourier_bases(x, self.M, ms)
            self.u_bases_deriv = lambda x: _deriv_fourier_bases(x, self.M, ms)
        elif u_bases == "chebyshev":
            ms = torch.arange(self.M, device=self.torch_dev)
            self.u_bases = lambda x: _chebyshev_bases(x, self.M, ms)
            self.u_bases_deriv = lambda x: _deriv_chebyshev_bases(x, self.M, ms)
        else:
            raise ValueError(f"Invalid u_bases: {u_bases}")

        # Last matrix used in fast gradient-field path (same as stream_func_coil)
        self.lct = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            device=self.torch_dev,
            dtype=torch.float32,
        )

    def _u_samples_1d(self, dtype: torch.dtype) -> torch.Tensor:
        """Half-open ``[u_min, u_max)`` grid with ``num_u`` steps, matching ``stream_func_coil`` theta sampling."""
        span = self.u_max - self.u_min
        u = torch.arange(
            self.u_min,
            self.u_max,
            span / self.num_u,
            device=self.torch_dev,
            dtype=dtype,
        )
        if u.numel() != self.num_u:
            u = torch.linspace(
                self.u_min,
                self.u_max,
                self.num_u + 1,
                device=self.torch_dev,
                dtype=dtype,
            )[:-1]
        return u

    def _v_samples_1d(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.linspace(self.v_min, self.v_max, self.num_v, device=self.torch_dev, dtype=dtype)

    # ----- coordinate maps (parallel to theta / z in stream_func_coil) -----

    def _u_coord_for_bases(self, u: torch.Tensor) -> torch.Tensor:
        if self._ub == "chebyshev":
            return 2.0 * (u - self.u_min) / (self.u_max - self.u_min) - 1.0
        # fourier (theta-style): map [u_min, u_max] -> [0, 2π)
        return (u - self.u_min) / (self.u_max - self.u_min) * (2.0 * torch.pi)

    def _du_chain(self) -> float:
        if self._ub == "chebyshev":
            return 2.0 / (self.u_max - self.u_min)
        return (2.0 * torch.pi) / (self.u_max - self.u_min)

    def _v_coord_for_bases(self, v: torch.Tensor) -> torch.Tensor:
        eta = 2.0 * (v - self.v_min) / (self.v_max - self.v_min) - 1.0
        if self._vb == "chebyshev":
            return eta
        # v fourier (z-style): eta in [-1, 1] passed to inner fourier like stream_func_coil
        return eta

    def _dv_chain(self) -> float:
        # deta/dv for both chebyshev and fourier-on-eta (v-style)
        return 2.0 / (self.v_max - self.v_min)

    # ----- surface geometry -----

    def _surface_positions(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.grad_surface.to_xyz(u, v)

    def _jacobian_dxyz_duv(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """``gradient_surfaces.surface.dxyz_duv`` returns ``(dxyz_du, dxyz_dv)``; stack to (..., 3, 2)."""
        out = self.grad_surface.dxyz_duv(u, v)
        if isinstance(out, tuple):
            dxyz_du, dxyz_dv = out
            return torch.stack([dxyz_du, dxyz_dv], dim=-1)
        return out

    def _surface_tangent_vectors(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        jac = self._jacobian_dxyz_duv(u, v)
        t_u = jac[..., 0]
        t_v = jac[..., 1]
        return t_u, t_v

    def _normal_vector_and_ginv(
        self, tangent_u: torch.Tensor, tangent_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mat = torch.stack([tangent_u, tangent_v], dim=-1)
        g = mat.mT @ mat
        det = g[..., 0, 0] * g[..., 1, 1] - g[..., 0, 1] * g[..., 1, 0]
        g_inv = torch.zeros_like(g)
        g_inv[..., 0, 0] = g[..., 1, 1]
        g_inv[..., 1, 1] = g[..., 0, 0]
        g_inv[..., 0, 1] = -g[..., 0, 1]
        g_inv[..., 1, 0] = -g[..., 1, 0]
        g_inv = g_inv / det[..., None, None]
        # Same convention as stream_func_coil: cross(t_z, t_theta) -> here cross(t_v, t_u)
        normal = torch.cross(tangent_v, tangent_u, dim=-1)
        return normal, g_inv

    def _stream_function_surface_gradient_bases(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uc = self._u_coord_for_bases(u)
        eta_v = self._v_coord_for_bases(v)

        ubd = self.u_bases_deriv(uc) * self._du_chain()
        vb = self.v_bases(eta_v)
        bu = vb[..., None] * ubd[..., None, :]
        combined_u = bu.reshape((*u.shape, -1))

        vbd = self.v_bases_deriv(eta_v) * self._dv_chain()
        ub = self.u_bases(uc)
        bv = vbd[..., None] * ub[..., None, :]
        combined_v = bv.reshape((*u.shape, -1))

        return combined_u, combined_v

    def _stream_function_surface_gradient(
        self, u: torch.Tensor, v: torch.Tensor, stream_coeffs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uc = self._u_coord_for_bases(u)
        eta_v = self._v_coord_for_bases(v)

        ubd = self.u_bases_deriv(uc) * self._du_chain()
        vb = self.v_bases(eta_v)
        bu = vb[..., None] * ubd[..., None, :]
        bu = bu.reshape((*u.shape, -1))
        dphi_du = bu @ stream_coeffs

        vbd = self.v_bases_deriv(eta_v) * self._dv_chain()
        ub = self.u_bases(uc)
        bv = vbd[..., None] * ub[..., None, :]
        bv = bv.reshape((*u.shape, -1))
        dphi_dv = bv @ stream_coeffs

        return dphi_du, dphi_dv

    def _stream_function(
        self, u: torch.Tensor, v: torch.Tensor, stream_coeffs: torch.Tensor
    ) -> torch.Tensor:
        uc = self._u_coord_for_bases(u)
        eta_v = self._v_coord_for_bases(v)
        vb = self.v_bases(eta_v)
        ub = self.u_bases(uc)
        combined = vb[..., None] * ub[..., None, :]
        combined = combined.reshape((*u.shape, -1))
        return combined @ stream_coeffs

    def _current_density_dS_bases(
        self, u: torch.Tensor, v: torch.Tensor, return_grad_phi: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_shape = u.shape
        v_shape = v.shape
        u_flat = u.reshape((-1,))
        v_flat = v.reshape((-1,))
        assert u_flat.shape == v_flat.shape

        bases_u, bases_v = self._stream_function_surface_gradient_bases(u_flat, v_flat)
        dphi_stack = torch.stack([bases_u, bases_v], dim=-1)

        t_u, t_v = self._surface_tangent_vectors(u_flat, v_flat)
        tangent_stack = torch.stack([t_u, t_v], dim=-1)

        n, g_inv = self._normal_vector_and_ginv(t_u, t_v)
        n_hat = n / n.norm(dim=-1, keepdim=True)

        grad_phi = tangent_stack[:, None, :] @ (g_inv[:, None, :, :] @ dphi_stack[..., None])
        grad_phi = grad_phi[..., 0]
        js = torch.cross(n_hat[:, None, :], grad_phi, dim=-1)

        ds_factor = n.norm(dim=-1)[:, None]

        js = js.reshape((*u_shape, js.shape[-2], 3))
        ds_factor = ds_factor.reshape(u_shape)

        if return_grad_phi:
            return js, ds_factor, grad_phi
        return js, ds_factor

    def _current_density_dS(
        self, u: torch.Tensor, v: torch.Tensor, stream_coeffs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_u, t_v = self._surface_tangent_vectors(u, v)
        tangent_stack = torch.stack([t_u, t_v], dim=-1)

        n, g_inv = self._normal_vector_and_ginv(t_u, t_v)
        n_hat = n / n.norm(dim=-1, keepdim=True)

        dphi_du, dphi_dv = self._stream_function_surface_gradient(u, v, stream_coeffs)
        dphi_stack = torch.stack([dphi_du, dphi_dv], dim=-1)

        grad_phi = tangent_stack @ (g_inv @ dphi_stack[..., None])
        grad_phi = grad_phi[..., 0]
        js = torch.cross(n_hat, grad_phi, dim=-1)
        ds_factor = n.norm(dim=-1)
        return js, ds_factor

    def evaluate_fields(
        self,
        coeffs: torch.Tensor,
        crds_bfield: torch.Tensor,
        crds_gfield: torch.Tensor,
        crds_efield: torch.Tensor,
        batch_size: Optional[int] = 2**3,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = crds_bfield.dtype
        mu0_over_4pi = 1e-7
        gshape = crds_gfield.shape[:-1]
        bshape = crds_bfield.shape[:-1]
        eshape = crds_efield.shape[:-1]

        crds_bfield = crds_bfield.view((-1, 3))
        crds_gfield = crds_gfield.view((-1, 3))
        crds_efield = crds_efield.view((-1, 3))

        u_surf = self._u_samples_1d(dtype)
        v_surf = self._v_samples_1d(dtype)
        dv = v_surf[1] - v_surf[0]
        if u_surf.numel() > 1:
            du = u_surf[1] - u_surf[0]
        else:
            du = torch.tensor(1.0, device=self.torch_dev, dtype=dtype)

        u_surf, v_surf = torch.meshgrid(u_surf, v_surf, indexing="ij")
        u_surf = u_surf.reshape(-1)
        v_surf = v_surf.reshape(-1)
        crds_surf = self._surface_positions(u_surf, v_surf)
        nsurf = crds_surf.shape[0]

        bfields = torch.zeros((crds_bfield.shape[0], 3), dtype=dtype, device=self.torch_dev)
        gfields = torch.zeros((crds_gfield.shape[0], 3), dtype=dtype, device=self.torch_dev)
        efields = torch.zeros((crds_efield.shape[0], 3), dtype=dtype, device=self.torch_dev)

        js, ds_factor = self._current_density_dS(u_surf, v_surf, coeffs)
        ds_factor = ds_factor * du * dv

        batch_size = nsurf if batch_size is None else batch_size
        for n1 in tqdm(range(0, nsurf, batch_size), "Evaluating Fields", disable=not verbose):
            n2 = min(n1 + batch_size, nsurf)

            diff = crds_bfield[:, None, :] - crds_surf[None, n1:n2, :]
            numer = torch.cross(js[None, n1:n2, :], diff, dim=-1) * ds_factor[None, n1:n2, None]
            denom = diff.norm(dim=-1, keepdim=True) ** 3
            bfields += mu0_over_4pi * (numer / denom).sum(dim=1)

            diff = crds_gfield[:, None, :] - crds_surf[None, n1:n2, :]
            denom1 = diff.norm(dim=-1, keepdim=True) ** 3
            numer1 = js[None, n1:n2, :] @ self.lct.mT
            numer1 = numer1 * ds_factor[None, n1:n2, None]
            denom2 = diff.norm(dim=-1, keepdim=True) ** 5
            numer2 = (js[None, n1:n2, :, None] * self.lct[None, None]).sum(dim=-2)
            numer2 = (numer2 * diff).sum(dim=-1)[..., None] * diff
            numer2 = numer2 * ds_factor[None, n1:n2, None]
            gfields += mu0_over_4pi * (numer1 / denom1 + 3 * numer2 / denom2).sum(dim=1)

            diff = crds_efield[:, None, :] - crds_surf[None, n1:n2, :]
            numer = js[None, n1:n2, :] * ds_factor[None, n1:n2, None]
            denom = diff.norm(dim=-1, keepdim=True)
            efields += mu0_over_4pi * (numer / denom).sum(dim=1)

        return bfields.reshape(*bshape, 3), gfields.reshape(*gshape, 3), efields.reshape(*eshape, 3)

    def build_winding_tolerance_matrix(
        self, num_u: int = 100, num_v: int = 100
    ) -> torch.Tensor:
        span = self.u_max - self.u_min
        u = torch.arange(self.u_min, self.u_max, span / num_u, device=self.torch_dev)
        if u.numel() != num_u:
            u = torch.linspace(self.u_min, self.u_max, num_u + 1, device=self.torch_dev)[:-1]
        v = torch.linspace(self.v_min, self.v_max, num_v, device=self.torch_dev)
        u, v = torch.meshgrid(u, v, indexing="ij")
        u = u.reshape(-1)
        v = v.reshape(-1)
        _, _, winding_tol = self._current_density_dS_bases(u, v, return_grad_phi=True)
        return winding_tol

    def build_current_boundary_matrix(self, verbose: bool = False) -> torch.Tensor:
        """Enforce ``J · t_v ≈ 0`` at ``v = v_min`` and ``v = v_max`` (u varying), mirroring z-boundaries in ``stream_func_coil``."""
        ncoeffs = self.M * self.K
        u = self._u_samples_1d(torch.float32)
        assert u.numel() == self.num_u
        
        # Sample edges
        u_edges, v_edges, normal_edges = self.grad_surface.sample_edges()

        jmat = torch.zeros((ncoeffs, len(u_edges)), dtype=torch.float32, device=self.torch_dev)
        one_hot = torch.zeros(ncoeffs, device=self.torch_dev)
        for c in tqdm(range(ncoeffs), "Building Field Matrix", disable=not verbose):
            one_hot.zero_()
            one_hot[c] = 1.0
            
            js, _ = self._current_density_dS(u_edges, v_edges, one_hot)
            jn = (js * normal_edges).sum(dim=-1)

            jmat[c, :] = jn

        return jmat.T

    def build_field_matrices_fast(
        self,
        crds_bfield: torch.Tensor,
        crds_gfield: torch.Tensor,
        crds_efield: torch.Tensor,
        batch_size: Optional[int] = 2**4,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_over_4pi = 1e-7
        gshape = crds_gfield.shape[:-1]
        bshape = crds_bfield.shape[:-1]
        eshape = crds_efield.shape[:-1]
        crds_bfield = crds_bfield.reshape((-1, 3))
        crds_gfield = crds_gfield.reshape((-1, 3))
        crds_efield = crds_efield.reshape((-1, 3))

        u = self._u_samples_1d(crds_bfield.dtype)
        assert u.numel() == self.num_u
        if u.numel() > 1:
            du = u[1] - u[0]
        else:
            du = torch.tensor(1.0, device=self.torch_dev, dtype=crds_bfield.dtype)

        v = torch.linspace(self.v_min, self.v_max, self.num_v, device=self.torch_dev, dtype=crds_bfield.dtype)
        dv = v[1] - v[0]
        u, v = torch.meshgrid(u, v, indexing="ij")
        u = u.reshape(-1)
        v = v.reshape(-1)
        crds_surf = self._surface_positions(u, v)
        nsurf = crds_surf.shape[0]

        batch_size = nsurf if batch_size is None else batch_size

        js, ds_factor = self._current_density_dS_bases(u, v)
        ds_factor = ds_factor * du * dv

        ncoeffs = self.M * self.K
        bfields = torch.zeros(
            (crds_bfield.shape[0], ncoeffs, 3), dtype=crds_bfield.dtype, device=self.torch_dev
        )
        gfields = torch.zeros(
            (crds_gfield.shape[0], ncoeffs, 3), dtype=crds_gfield.dtype, device=self.torch_dev
        )
        afields = torch.zeros(
            (crds_efield.shape[0], ncoeffs, 3), dtype=crds_bfield.dtype, device=self.torch_dev
        )

        for n1 in tqdm(range(0, nsurf, batch_size), "Building Field Matrices", disable=not verbose):
            n2 = min(n1 + batch_size, nsurf)

            diff_mgnt = crds_bfield[:, None, None, :] - crds_surf[None, n1:n2, None, :]
            numer_mgnt = (
                torch.cross(js[None, n1:n2], diff_mgnt, dim=-1) * ds_factor[None, n1:n2, None, None]
            )
            denom_mgnt = diff_mgnt.norm(dim=-1, keepdim=True) ** 3
            bfields += mu0_over_4pi * (numer_mgnt / denom_mgnt).sum(dim=1)

            diff_grdnt = crds_gfield[:, None, :] - crds_surf[None, n1:n2, :]
            denom1 = diff_grdnt.norm(dim=-1) ** 3
            numer1 = (self.lct[None, None, :, :] @ js[n1:n2, :, :, None])[..., 0]
            denom2 = diff_grdnt.norm(dim=-1) ** 5
            numer2 = (self.lct[None, None, :, :] @ diff_grdnt[..., None])[..., 0]
            numer2 = (js[None, n1:n2, :, :] * numer2[:, :, None, :]).sum(dim=-1)
            numer2 = (numer2[..., None] * diff_grdnt[:, :, None, :]) * 3
            integrand = (
                (numer1 / denom1[:, :, None, None]) + (numer2 / denom2[:, :, None, None])
            ) * ds_factor[None, n1:n2, None, None]
            gfields += mu0_over_4pi * integrand.sum(dim=1)

            diff_afld = crds_efield[:, None, None, :] - crds_surf[None, n1:n2, None, :]
            numer_afld = js[None, n1:n2, :] * ds_factor[None, n1:n2, None, None]
            denom_afld = diff_afld.norm(dim=-1, keepdim=True)
            afields += mu0_over_4pi * (numer_afld / denom_afld).sum(dim=1)

        bfields = bfields.reshape((*bshape, ncoeffs, 3))
        gfields = gfields.reshape((*gshape, ncoeffs, 3))
        efields = -afields.reshape((*eshape, ncoeffs, 3))
        return bfields, gfields, efields

    def build_field_matrices(
        self,
        crds_bfield: torch.Tensor,
        crds_gfield: torch.Tensor,
        crds_efield: torch.Tensor,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.build_field_matrices_fast(crds_bfield, crds_gfield, crds_efield, verbose=verbose)

    def build_magnetic_energy_matrix(
        self, batch_size: Optional[int] = 2**4, verbose: bool = True
    ) -> torch.Tensor:
        ncoeffs = self.M * self.K
        mu0_over_4pi = 1e-7

        u = self._u_samples_1d(torch.float32)
        if u.numel() > 1:
            du = u[1] - u[0]
        else:
            du = torch.tensor(1.0, device=self.torch_dev, dtype=torch.float32)

        v_line = torch.linspace(self.v_min, self.v_max, self.num_v, device=self.torch_dev, dtype=torch.float32)
        if v_line.numel() > 1:
            dv = v_line[1] - v_line[0]
        else:
            dv = torch.tensor(1.0, device=self.torch_dev, dtype=torch.float32)
        u, v = torch.meshgrid(u, v_line, indexing="ij")
        u = u.reshape(-1)
        v = v.reshape(-1)
        crds_surf = self._surface_positions(u, v)
        nsurf = crds_surf.shape[0]

        js, ds_factor = self._current_density_dS_bases(u, v)
        ds_factor = ds_factor * du * dv

        w = torch.zeros((ncoeffs, ncoeffs), dtype=torch.float32, device=self.torch_dev)
        eps = 1e-3**2

        bs = nsurf if batch_size is None else batch_size
        for n1 in tqdm(range(0, nsurf, bs), "Building Energy Matrix", disable=not verbose):
            n2 = min(n1 + bs, nsurf)
            diff = crds_surf[:, None, :] - crds_surf[None, n1:n2, :]
            inv_nrm = 1.0 / (diff.square().sum(dim=-1) + eps).sqrt()
            inv_nrm = inv_nrm.T
            integral = einsum(
                js[None,] * ds_factor[None, :, None, None] * inv_nrm[:, :, None, None],
                js[n1:n2, None] * ds_factor[n1:n2, None, None, None],
                "B N Ci d, B N Co d -> Ci Co",
            )
            w += 0.5 * mu0_over_4pi * integral

        inductance = 2 * w / (1.0**2)
        return inductance

    def stream_to_contour(
        self,
        stream_coeffs: torch.Tensor,
        num_u: int = 100,
        num_v: int = 200,
        dstream: Optional[float] = None,
    ) -> tuple[list[torch.Tensor], float]:
        v = torch.linspace(self.v_min, self.v_max, num_v, device=self.torch_dev)
        u = torch.linspace(self.u_min, self.u_max, num_u, device=self.torch_dev)
        uu, vv = torch.meshgrid(u, v, indexing="ij")
        stream = self._stream_function(uu, vv, stream_coeffs)

        if dstream is None:
            dstream = (stream.max() - stream.min()).item() / 100.0
            print(f"Stream min = {stream.min():1.2e}")
            print(f"Stream max = {stream.max():1.2e}")
            print(f"dstream = {dstream:1.2e}")

        levels = torch.arange(stream.min(), stream.max(), dstream) + dstream / 2
        u_np = u.cpu().numpy()
        v_np = v.cpu().numpy()
        xyz_contours: list[torch.Tensor] = []
        theta_z_contours: list[torch.Tensor] = []

        for level in levels:
            stream_cat = torch.cat([stream[-1:, :], stream], dim=0)
            contours_rc = measure.find_contours(stream_cat.cpu().numpy(), level=float(level))

            for rc in contours_rc:
                u_c = interp1d(np.arange(len(u_np)), u_np, kind="linear", fill_value="extrapolate")(rc[:, 0])
                v_c = interp1d(np.arange(len(v_np)), v_np, kind="linear", fill_value="extrapolate")(rc[:, 1])
                u_c = torch.from_numpy(u_c).to(self.torch_dev)
                v_c = torch.from_numpy(v_c).to(self.torch_dev)
                xyz_contour = self._surface_positions(u_c, v_c).cpu()
                uv_contour = torch.stack([u_c, v_c], dim=-1).type(torch.float32)
                theta_z_contours.append(uv_contour.reshape(-1, 2))
                xyz_contour = xyz_contour.type(torch.float32)
                xyz_contours.append(xyz_contour.reshape(-1, 3))

        flipped = 0
        for i in range(len(xyz_contours)):
            uv_surf = theta_z_contours[i][:1]
            j_surf, _ = self._current_density_dS(uv_surf[:, 0], uv_surf[:, 1], stream_coeffs)
            tangent = xyz_contours[i][1] - xyz_contours[i][0]
            sign = torch.sign(j_surf.unsqueeze(0).cpu() @ tangent)
            if sign < 0:
                flipped += 1
                xyz_contours[i] = xyz_contours[i].flip(dims=[0])
        print(f"Flipped {flipped}/{len(xyz_contours)} contours")
        return xyz_contours, dstream

    def show_countour(
        self,
        stream_coeffs: torch.Tensor,
        num_u: int = 100,
        num_v: int = 50,
        dstream: Optional[float] = None,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        if fig is None or ax is None:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, projection="3d")

        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_zlabel("z (cm)")
        ax.set_aspect("equal")

        xyz_contours, _d = self.stream_to_contour(stream_coeffs, num_u, num_v, dstream=dstream)
        for xyz_c in xyz_contours:
            ax.plot(
                xyz_c[..., 0] * 1e2,
                xyz_c[..., 1] * 1e2,
                xyz_c[..., 2] * 1e2,
                color="red",
                alpha=0.3,
            )
        return fig, ax

    def show_current_density(
        self,
        stream_coeffs: torch.Tensor,
        num_u: int = 100,
        num_v: int = 50,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
    ):
        if fig is None or ax is None:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, projection="3d")

        u = torch.linspace(self.u_min, self.u_max, num_u, device=self.torch_dev)
        v = torch.linspace(self.v_min, self.v_max, num_v, device=self.torch_dev)
        u, v = torch.meshgrid(u, v, indexing="ij")
        crds = self._surface_positions(u, v)
        js, _ = self._current_density_dS(u, v, stream_coeffs)

        crds = crds.reshape(-1, 3).cpu() * 1e2
        js = js.reshape(-1, 3).cpu()
        mag = js.norm(dim=-1, keepdim=True)
        js_dir = js / mag
        mag /= mag.max()
        js = js_dir * (mag**0.5)
        ax.quiver(
            crds[..., 0],
            crds[..., 1],
            crds[..., 2],
            js[..., 0],
            js[..., 1],
            js[..., 2],
            length=2,
            alpha=0.4,
            color="red",
        )
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_zlabel("z (cm)")
        ax.set_aspect("equal")
        return fig, ax

    def show_design(
        self,
        coeffs: torch.Tensor,
        num_u: int = 100,
        num_v: int = 50,
        body_surf: Optional[torch.Tensor] = None,
        show_1d: bool = False,
        colorbar: bool = True,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        if fig is None or ax is None:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, projection="3d")

        v = torch.linspace(self.v_min, self.v_max, num_v, device=self.torch_dev)
        u = torch.linspace(self.u_min, self.u_max, num_u, device=self.torch_dev)
        u, v = torch.meshgrid(u, v, indexing="ij")
        crds = self._surface_positions(u, v)
        xs, ys, zs_map = crds[..., 0], crds[..., 1], crds[..., 2]
        vals = self._stream_function(u, v, coeffs)

        xs, ys, zs_map, vals = xs.cpu(), ys.cpu(), zs_map.cpu(), vals.cpu()
        alpha = 0.5 if body_surf is not None else 1.0
        norm = plt.Normalize(vmin=-vals.abs().max(), vmax=vals.abs().max())
        colormap = cm.berlin
        colors = colormap(norm(vals))
        sm = cm.ScalarMappable(norm=norm, cmap=colormap)
        ax.plot_surface(
            xs * 1e2,
            ys * 1e2,
            zs_map * 1e2,
            facecolors=colors,
            rcount=zs_map.shape[0],
            ccount=zs_map.shape[1],
            alpha=alpha,
            shade=False,
            linewidth=0,
            edgecolor="none",
        )
        ax.set_aspect("equal")

        if colorbar:
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Stream Function Value")

        if body_surf is not None:
            ax.plot_surface(
                body_surf[..., 0].cpu(),
                body_surf[..., 1].cpu(),
                body_surf[..., 2].cpu(),
                color="navajowhite",
                alpha=1.0,
                shade=True,
                linewidth=0,
                edgecolor="none",
            )

        return fig, ax

