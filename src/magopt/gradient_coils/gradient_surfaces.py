import torch
from typing import Optional
from ..bspline import BSpline1D

class surface:
    
    def __init__(self,
                 device: torch.device = torch.device('cpu')):
        self.device = device
    
    def to_xyz(self, 
               u: torch.Tensor, 
               v: torch.Tensor) -> torch.Tensor:
        """
        Returns the surface position for the given u and v parameters.
        
        Args
        ----
        u : torch.Tensor
            shape (...) representing the u coordinates.
        v : torch.Tensor
            shape (...) representing the v coordinates. 
            
        Returns
        -------
        xyz : torch.Tensor
            shape (..., 3) representing the surface position.
        """
        raise NotImplementedError
    
    def dxyz_duv(self, 
                 u: torch.Tensor, 
                 v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the derivative of the surface position with respect to the u and v parameters.
        
        Args
        ----
        u : torch.Tensor
            shape (...) representing the u coordinates.
        v : torch.Tensor
            shape (...) representing the v coordinates. 
            
        Returns
        -------
        dxyz_du : torch.Tensor
            shape (..., 3) representing the derivative of the surface position with respect to the u parameter.
        dxyz_dv : torch.Tensor
            shape (..., 3) representing the derivative of the surface position with respect to the v parameter.
        """
        raise NotImplementedError
    
    def sample_edges(self,):
        """
        Returns a list of u, v coordinates at the edges of the surface,
        along with the normal vector pointing out of the surface.
        
        This is to help enforce current density continuity at the surface edges.
        
        Returns
        -------
        u_edges : torch.Tensor
            shape (N) representing the u coordinates at the edges of the surface.
        v_edges : torch.Tensor
            shape (N) representing the v coordinates at the edges of the surface.
        normal_edges : torch.Tensor
            shape (N, 3) representing the normal vector pointing out of the surface at the edges.
        """
        raise NotImplementedError
    
class elliptical_frustum(surface):
    """
    Elliptical frustum surface.
    
    The u parameter represents the angle around the z-axis.
    The v parameter represents the position along the z-axis.
    """
    
    def __init__(self, 
                 zs_spline: torch.Tensor,
                 as_spline: torch.Tensor,
                 bs_spline: Optional[torch.Tensor] = None,
                 lamda_spline: float = 1e-2,
                 **kwargs):
        """
        Args
        ----
        zs_spline : torch.Tensor
            shape (K,) representing the z positions of the spline interpolation points.
        as_spline : torch.Tensor
            shape (K,) representing the the x-radii positions of the spline interpolation points.
        bs_spline : torch.Tensor
            shape (K,) representing the the y-radii positions of the spline interpolation points.
        lamda_spline : float, optional
            Regularization parameter for the spline fitting.
        **kwargs : dict, optional
            Additional keyword arguments for general surface constructor
        """
        # Consts
        self.zmin = zs_spline.min().item()
        self.zmax = zs_spline.max().item()
        
        # Convert z to v values
        vs_spline = (zs_spline - self.zmin) / (self.zmax - self.zmin)
        
        # Build spline functions
        self.as_spline = BSpline1D(len(as_spline), vs_spline.min().item(), vs_spline.max().item(),
                                   boundary="clamp", dtype=as_spline.dtype, device=as_spline.device)
        self.as_spline.fit_lstsq(vs_spline, as_spline, lamda_spline)
        if bs_spline is None:
            self.bs_spline = self.as_spline
        else:
            self.bs_spline = BSpline1D(len(bs_spline), vs_spline.min().item(), vs_spline.max().item(),
                                       boundary="clamp", dtype=bs_spline.dtype, device=bs_spline.device)
            self.bs_spline.fit_lstsq(vs_spline, bs_spline, lamda_spline)
            
        # Call general surface constructor
        super().__init__(**kwargs)
            
    def to_xyz(self, 
               u: torch.Tensor, 
               v: torch.Tensor) -> torch.Tensor:
        a = self.as_spline(v)
        b = self.bs_spline(v)
        return torch.stack([
            a * torch.cos(2 * torch.pi * u),
            b * torch.sin(2 * torch.pi * u),
            v * self.zmax + (1 - v) * self.zmin,
        ], dim=-1)
    
    def dxyz_duv(self, 
                 u: torch.Tensor, 
                 v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.as_spline(v)
        b = self.bs_spline(v)
        a_prime = self.as_spline.dy(v)
        b_prime = self.bs_spline.dy(v)
        dxyz_dv = torch.stack([
            a_prime * torch.cos(2 * torch.pi * u),
            b_prime * torch.sin(2 * torch.pi * u),
            torch.ones_like(u) * (self.zmax - self.zmin),
        ], dim=-1)
        dxyz_du = torch.stack([
            -2 * torch.pi * a * torch.sin(2 * torch.pi * u),
            2 * torch.pi * b * torch.cos(2 * torch.pi * u),
            torch.zeros_like(u),
        ], dim=-1)
        return dxyz_du, dxyz_dv
    
    def sample_edges(self,
                     num_pts: int = 100):
        u_lin = torch.linspace(0, 1, num_pts//2)
        u_edges = torch.cat([u_lin, u_lin])
        v_edges = torch.cat([u_lin*0, u_lin*0 + 1])
        _, normal_edges = self.dxyz_duv(u_edges, v_edges)
        normal_edges /= normal_edges.norm(dim=-1, keepdim=True)
        return u_edges, v_edges, normal_edges
        
class planar_surface(surface):
    """
    Planar surface.
    
    The u parameter variation about the first coordinate axis (see below)
    The v parameter variation about the second coordinate axis (see below)
    """
    
    def __init__(self, 
                 u_axis: torch.Tensor = torch.tensor([1, 0, 0]),
                 v_axis: torch.Tensor = torch.tensor([0, 1, 0]),
                 center: torch.Tensor = torch.tensor([0, 0, 0]),
                 width_u: float = 1.0,
                 width_v: float = 1.0,
                 **kwargs):
        """
        Args
        ----
        u_axis : torch.Tensor
            shape (3,) representing the first coordinate axis.
        v_axis : torch.Tensor
            shape (3,) representing the second coordinate axis.
        **kwargs : dict, optional
            Additional keyword arguments for general surface constructor
        """
        # Call general surface constructor
        super().__init__(**kwargs)
        
        # Store axes
        self.u_axis = u_axis.to(self.device)
        self.v_axis = v_axis.to(self.device)
        self.center = center.to(self.device)
        self.w_u = width_u
        self.w_v = width_v
        
    def to_xyz(self, 
               u: torch.Tensor, 
               v: torch.Tensor) -> torch.Tensor:
        u_cent = u - 0.5
        v_cent = v - 0.5
        return self.center + self.w_u * self.u_axis * u_cent[..., None] + self.w_v * self.v_axis * v_cent[..., None]

    def dxyz_duv(self, 
                 u: torch.Tensor, 
                 v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ru = torch.ones_like(u)[..., None] * self.u_axis * self.w_u
        rv = torch.ones_like(v)[..., None] * self.v_axis * self.w_v
        return ru, rv
    
    def sample_edges(self,
                     num_pts: int = 100):
        u_lin = torch.linspace(0, 1, num_pts//4, device=self.device)
        v_lin = torch.linspace(0, 1, num_pts//4, device=self.device)
        
        # vmin = 0, u line
        u1 = u_lin
        v1 = v_lin*0
        ru1, rv1 = self.dxyz_duv(u1, v1)
        n1 = rv1 / rv1.norm(dim=-1, keepdim=True)
        
        # vmax = 1, u line
        u2 = u_lin
        v2 = v_lin*0 + 1
        ru2, rv2 = self.dxyz_duv(u2, v2)
        n2 = rv2 / rv2.norm(dim=-1, keepdim=True)
        
        # umax = 1, v line
        u3 = u_lin*0 + 1
        v3 = v_lin
        ru3, rv3 = self.dxyz_duv(u3, v3)
        n3 = ru3 / ru3.norm(dim=-1, keepdim=True)
        
        # umin = 0, v line
        u4 = u_lin*0
        v4 = v_lin
        ru4, rv4 = self.dxyz_duv(u4, v4)
        n4 = ru4 / ru4.norm(dim=-1, keepdim=True)
        
        # Combine
        u_edges = torch.cat([u1, u2, u3, u4])
        v_edges = torch.cat([v1, v2, v3, v4])
        normal_edges = torch.cat([n1, n2, n3, n4], dim=0)
        return u_edges, v_edges, normal_edges
    
class planar_curved_surface(surface):
    """
    Planar curved surface.
    
    The u parameter variation about the first coordinate axis (see below)
    The v parameter variation about the second coordinate axis (see below)
    """
    
    def __init__(self,
                 u_axis: torch.Tensor = torch.tensor([1, 0, 0]),
                 v_axis: torch.Tensor = torch.tensor([0, 1, 0]),
                 center: torch.Tensor = torch.tensor([0, 0, 0]),
                 width_u: float = 1.0,
                 width_v: float = 1.0,
                 height_curve: float = 1.0,
                 poly_degree: int = 6,
                 **kwargs):
        """
        Args
        ----
        u_axis : torch.Tensor
            shape (3,) representing the first coordinate axis.
        v_axis : torch.Tensor
            shape (3,) representing the second coordinate axis.
        center : torch.Tensor
            shape (3,) representing the center of the surface.
        width_u : float
            shape (3,) representing the width of the surface in the u direction.
        width_v : float
            shape (3,) representing the width of the surface in the v direction.
        height_curve : float
            shape (3,) representing the height of the surface in the v direction.
        **kwargs : dict, optional
            Additional keyword arguments for general surface constructor
        """
        # Call general surface constructor
        super().__init__(**kwargs)
        assert poly_degree % 2 == 0, "Polynomial degree must be even"
        
        # Store axes
        self.u_axis = u_axis.to(self.device)
        self.v_axis = v_axis.to(self.device)
        self.center = center.to(self.device)
                
        # Compute orthogonal axis
        o_axis = torch.cross(self.u_axis, self.v_axis, dim=-1)
        o_axis /= o_axis.norm(dim=-1, keepdim=True)
        self.o_axis = o_axis
        
        # Store widths
        self.w_u = width_u
        self.w_v = width_v
        self.w_o = height_curve
        self.p = poly_degree
        
    def to_xyz(self, 
               u: torch.Tensor, 
               v: torch.Tensor) -> torch.Tensor:
        # v axis
        v_cent = v - 0.5
        crds = self.w_v * self.v_axis * v_cent[..., None]
        
        # u axis 
        u_cent = u - 0.5
        crds += self.u_axis * self.w_u * u_cent[..., None]
        
        # curvature along u axis
        hu = (2 * u - 1)**(self.p)
        crds += self.o_axis * self.w_o * hu[..., None]
        
        # Add offset
        crds += self.center
        return crds
        
    def dxyz_duv(self, 
                 u: torch.Tensor, 
                 v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Derivative
        ru = torch.ones_like(u)[..., None] * self.u_axis * self.w_u
        rv = torch.ones_like(v)[..., None] * self.v_axis * self.w_v
        
        # Add curvature term
        ru += self.o_axis * self.w_o * 2 * self.p * (2 * u[..., None] - 1) ** (self.p - 1)
        
        return ru, rv

    def sample_edges(self,
                     num_pts: int = 100):
        u_lin = torch.linspace(0, 1, num_pts//4, device=self.device)
        v_lin = torch.linspace(0, 1, num_pts//4, device=self.device)
        
        # vmin = 0, u line
        u1 = u_lin
        v1 = v_lin*0
        ru1, rv1 = self.dxyz_duv(u1, v1)
        n1 = rv1 / rv1.norm(dim=-1, keepdim=True)
        
        # vmax = 1, u line
        u2 = u_lin
        v2 = v_lin*0 + 1
        ru2, rv2 = self.dxyz_duv(u2, v2)
        n2 = rv2 / rv2.norm(dim=-1, keepdim=True)
        
        # umax = 1, v line
        u3 = u_lin*0 + 1
        v3 = v_lin
        ru3, rv3 = self.dxyz_duv(u3, v3)
        n3 = ru3 / ru3.norm(dim=-1, keepdim=True)   
        
        # umin = 0, v line
        u4 = u_lin*0
        v4 = v_lin
        ru4, rv4 = self.dxyz_duv(u4, v4)
        n4 = ru4 / ru4.norm(dim=-1, keepdim=True)
        
        # Combine
        u_edges = torch.cat([u1, u2, u3, u4])
        v_edges = torch.cat([v1, v2, v3, v4])
        normal_edges = torch.cat([n1, n2, n3, n4], dim=0)
        return u_edges, v_edges, normal_edges