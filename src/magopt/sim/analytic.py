import torch

from tqdm import tqdm
from typing import Optional

from .elip import EllipELookup, EllipKLookup

MU0 = 4e-7 * torch.pi # T*m/A
EPSILON_STABILITY = 1e-9 # Small value to avoid division by zero in calculations

def _transform_coordinates(crds: torch.Tensor,
                           center: torch.Tensor,
                           normal: torch.Tensor,
                           flip_order: bool = False) -> torch.Tensor:
    """
    Rotates the spatial coordinates the z-axis maps to 'normal',
    and the center of the coordinate system is at 'center'
    
    Args:
    -----
    crds : torch.Tensor
        Spatial coordinates with shape (..., 3) in units [m]
    center : torch.Tensor
        Center of the coordinate system in units [m] with shape (..., 3)
    normal : torch.Tensor
        Normal vector of the coordinate system in units [m] with shape (..., 3)
    flip_order : bool
        If True, the translation/rotation order is flipped.

    Returns:
    --------
    transformed_crds : torch.Tensor
        Transformed spatial coordinates with shape (..., 3) in units [m]
    """
    # Check normal
    assert (normal.norm(dim=-1) - 1.0).abs().max() < 1e-5, "Normal vector must be a unit vector"
    
    # Move to same device
    center = center.to(crds.device)
    normal = normal.to(crds.device)
    
    # First construct an arbitrary rotation matrix that maps the z-axis to 'normal's
    basis_vec = torch.zeros_like(crds)
    basis_vec[..., 0] = 1.0
    # if (basis_vec @ normal).abs() > 0.999:
    #     basis_vec = torch.tensor([0, 1, 0], device=crds.device, dtype=crds.dtype)
    xp = torch.cross(basis_vec, normal, dim=-1)
    xp = xp / xp.norm(dim=-1, keepdim=True)
    yp = torch.cross(normal, xp, dim=-1)
    zp = normal + xp * 0
    Rot = torch.stack([xp, yp, zp], dim=-1)
    
    # Transform coordinates
    if flip_order:
        transformed_crds = (crds[..., None] * Rot.mT).sum(dim=-2) + center
    else:
        transformed_crds = ((crds - center)[..., None] * Rot).sum(dim=-2)
    return transformed_crds

def calc_mag_potential_loop(spatial_crds: torch.Tensor, 
                            R: float,
                            center: Optional[torch.Tensor] = torch.zeros(3),
                            normal: Optional[torch.Tensor] = torch.tensor([0, 0, 1]),
                            ellipe: Optional[torch.nn.Module] = EllipELookup(),
                            ellipk: Optional[torch.nn.Module] = EllipKLookup()) -> torch.Tensor:
    """
    Calculates the magnetic potential at any point in space due to 
    a circular loop of current

    Args:
    -----
    spatial_crds : torch.Tensor
        Spatial coordinates with shape (..., 3) in units [m]
    R : float
        Radius of the circular loop in units [m]
    center : torch.Tensor
        Center of the circular loop in units [m]
    normal : torch.Tensor
        Normal vector of the circular loop in units [m]

    Returns:
    --------
    A : torch.Tensor
        The magnetic potential with shape (..., 3) in units [T*m]
    """
    # Transform coordinates
    spatial_crds = _transform_coordinates(spatial_crds, center, normal)
    
    # Convert to cylindrical coordinates
    rho = (spatial_crds[..., 0] ** 2 + spatial_crds[..., 1] ** 2 + EPSILON_STABILITY).sqrt()
    z = spatial_crds[..., 2]
    
    # # Compute the magnetic potential in azimuthal direction
    # k = ((4 * R * rho) / ((rho + R) ** 2 + z ** 2)).sqrt()
    # const = MU0 / (2 * torch.pi * k * (R * rho).sqrt())
    # inner = (1 - k.square() / 2) * ellipk(k.square()) \
    #                              - ellipe(k.square())
    k = ((4 * R * rho) / (R ** 2 + rho ** 2 + z ** 2 + 2 * R * rho)).sqrt()
    const = MU0 * ((R / rho) ** 0.5) * 2  / (k * 2 * torch.pi)
    inner = (1 - k.square() / 2) * ellipk(k.square()) - ellipe(k.square())
    A_azimuthal = const * inner

    # Convert to cartesian coordinates
    azimuthal_vec = torch.stack((-spatial_crds[..., 1], spatial_crds[..., 0], torch.zeros_like(spatial_crds[..., 0])), dim=-1)
    A = A_azimuthal[..., None] * azimuthal_vec / rho[..., None]

    return A

def calc_bfield_loop(spatial_crds: torch.Tensor,
                     R: float,
                     center: Optional[torch.Tensor] = torch.zeros(3),
                     normal: Optional[torch.Tensor] = torch.tensor([0, 0, 1]),
                     ellipe: Optional[torch.nn.Module] = EllipELookup(),
                     ellipk: Optional[torch.nn.Module] = EllipKLookup()) -> torch.Tensor:
    """
    Calculates the magnetic field at a point in space due to
    a circular loop of current lying in the XY plane at Z=0.
    
    Args:
    -----
    spatial_crds : torch.Tensor
        Spatial coordinates with shape (..., 3) in units [m]
    R : float
        Radius of the circular loop in units [m] with shape (...,)
    center : torch.Tensor
        Center of the circular loop in units [m] with shape (..., 3)
    normal : torch.Tensor
        Normal vector of the circular loop in units [m] with shape (..., 3)

    Returns:
    --------
    bfield : torch.Tensor
        The magnetic field with shape (..., 3) in units [T]
    """
    # Transform coordinates
    spatial_crds = _transform_coordinates(spatial_crds, center, normal)
    
    # Convert to cylindrical coordinates
    rho = (spatial_crds[..., 0] ** 2 + spatial_crds[..., 1] ** 2).sqrt()
    z = spatial_crds[..., 2]
    
    # Compute constant/shared terms
    const = MU0 / (2 * torch.pi * ((rho + R).square() + z.square()).sqrt())
    m = ((4 * R * rho) / ((rho + R).square() + z.square()))
    K_term = ellipk(m)
    E_term = ellipe(m)

    # Bfield in radial direction
    B_rho = const * (z / (rho + EPSILON_STABILITY)) * (-K_term + \
        (R ** 2 + rho.square() + z.square()) * E_term / ((rho - R).square() + z.square()))
    
    # Bfield in z direction
    B_z = const * (K_term + \
        (R ** 2 - rho.square() - z.square()) * E_term / ((rho - R).square() + z.square()))
    
    # Get normal vectors in radial and z directions
    rho_hat = spatial_crds.clone()
    rho_hat[..., 2] = 0
    rho_hat /= rho_hat.norm(dim=-1, keepdim=True)
    z_hat = torch.zeros_like(spatial_crds)
    z_hat[..., 2] = 1
    
    # Convert Bfield to cartesian coordinates
    bfield = B_rho[..., None] * rho_hat + \
             B_z[..., None] * z_hat
             
    return bfield

def calc_inductance_loop(radius_loop: float,
                         radius_wire: Optional[float] = 0.5e-3) -> float:
    """
    Calculates the inductance of a circular loop of current.
    
    Args:
    -----
    radius_loop : float
        Radius of the circular loop in units [m]
    radius_wire : float
        radius of the wire in units [m]
        
    Returns:
    --------
    inductance : float
        Inductance of the loop in units [H]
    """
    return MU0 * radius_loop * (torch.log(8 * radius_loop / radius_wire) - 2)

def calc_mutual_inductance_pair(radius_loop1: float,
                                radius_loop2: float,
                                distance: float,
                                ellipe: Optional[torch.nn.Module] = EllipELookup(),
                                ellipk: Optional[torch.nn.Module] = EllipKLookup()) -> float:
    """
    Calculates the mutual inductance between two circular loops of current.
    
    Args:
    -----
    radius_loop1 : float
        Radius of the first circular loop in units [m]
    radius_loop2 : float
        Radius of the second circular loop in units [m]
    distance : float
        Distance between the centers of the two loops in units [m]
        
    Returns:
    --------
    mutual_inductance : float
        Mutual inductance between the two loops in units [H]
    """
    k_squared = 4 * radius_loop1 * radius_loop2
    k_squared /= (radius_loop1 + radius_loop2) ** 2 + distance ** 2
    k = k_squared ** 0.5

    mutual_inductance = (2 / k - k) * ellipk(k_squared) - (2 / k) * ellipe(k_squared)
    mutual_inductance *= MU0 * ((radius_loop1 * radius_loop2) ** 0.5)
    
    return mutual_inductance

def calc_inductance_matrix(radii: torch.Tensor,
                           positions: torch.Tensor,
                           radius_wire: Optional[float] = 0.5e-3,
                           ellipe: Optional[torch.nn.Module] = EllipELookup(),
                           ellipk: Optional[torch.nn.Module] = EllipKLookup()) -> torch.Tensor:
    """
    Computes inductance matrix for a set of circular loops defined by their radii and positions.
    
    The inductance then becomes:
    inductance = n.T L n
    
    where n is the number of turns per segment
    
    Args:
    -----
    radii : torch.Tensor
        Tensor of radii for each circular loop with shape (N,) 
    positions : torch.Tensor
        Tensor of positions for each circular loop with shape (N,)
    radius_wire : float, optional
        Radius of the wire used to calculate inductance [m]
        
    Returns:
    --------
    L : torch.Tensor
        Inductance matrix for the circular loops with shape (N, N)
    """
    # Consts
    N = radii.shape[0]
    assert positions.shape[0] == N, "Positions must match the number of radii"
    
    L = calc_mutual_inductance_pair(radii[:, None], radii[None, :],
                                    distance=(positions[:, None] - positions[None, :]).abs(),
                                    ellipe=ellipe,
                                    ellipk=ellipk)
    idxs_diags = torch.arange(N, device=radii.device)
    L[idxs_diags, idxs_diags] = calc_inductance_loop(radii, radius_wire=radius_wire)
    return L

def _dKdm(m, K, E, eps=1e-12):
    # dK/dm = (E/(m(1-m)) - K/m)/2
    m = m.clamp(eps, 1.0 - eps)
    return 0.5 * (E / (m * (1.0 - m)) - K / m)

def _dEdm(m, K, E, eps=1e-12):
    # dE/dm = (E - K) / (2m)
    m = m.clamp(eps, 1.0 - eps)
    return 0.5 * (E - K) / m

def calc_bfield_loop_jacobian(
    spatial_crds: torch.Tensor,
    R: float,
    center: Optional[torch.Tensor] = torch.zeros(3),
    normal: Optional[torch.Tensor] = torch.tensor([0, 0, 1]),
    ellipe: Optional[torch.nn.Module] = EllipELookup(),
    ellipk: Optional[torch.nn.Module] = EllipKLookup()) -> torch.Tensor:
    """
    Analytic Jacobian of the B-field from a circular loop in the XY plane at z=z_ofs.
    Returns (..., 3, 3) with entries dB_i/dx_j in Cartesian coords.
    
    Much thanks to Chat-GPT 5o!
    """
    # Transform coordinates
    spatial_crds = _transform_coordinates(spatial_crds, center, normal)

    if ellipe is None or ellipk is None:
        raise ValueError("Provide elliptic integral modules ellipe and ellipk.")

    x = spatial_crds[..., 0]
    y = spatial_crds[..., 1]
    z = spatial_crds[..., 2]

    rho = torch.sqrt(x*x + y*y + EPSILON_STABILITY)  # stabilized radius
    cosphi = x / (rho + EPSILON_STABILITY)
    sinphi = y / (rho + EPSILON_STABILITY)

    A  = (rho + R)**2 + z**2                            # (rho+R)^2 + z^2
    dA_drho = 2.0 * (rho + R)
    dA_dz   = 2.0 * z

    Delta  = (rho - R)**2 + z**2                        # (rho-R)^2 + z^2
    dD_drho = 2.0 * (rho - R)
    dD_dz   = 2.0 * z

    S   = R**2 + rho**2 + z**2
    dS_drho = 2.0 * rho
    dS_dz   = 2.0 * z

    S2  = R**2 - rho**2 - z**2
    dS2_drho = -2.0 * rho
    dS2_dz   = -2.0 * z

    # modulus
    m = (4.0 * R * rho) / A
    K = ellipk(m)
    E = ellipe(m)

    dK_dm = _dKdm(m, K, E, eps=EPSILON_STABILITY)
    dE_dm = _dEdm(m, K, E, eps=EPSILON_STABILITY)

    # dm/d(rho), dm/dz
    # m = (4 R rho) / A
    dm_drho = (4.0 * R * A - 4.0 * R * rho * dA_drho) / (A * A)
    dm_dz   = (-4.0 * R * rho * dA_dz) / (A * A)

    dK_drho = dK_dm * dm_drho
    dK_dz   = dK_dm * dm_dz
    dE_drho = dE_dm * dm_drho
    dE_dz   = dE_dm * dm_dz

    # C = MU0 / (2π √A)
    C = MU0 / (2.0 * torch.pi * torch.sqrt(A))
    # dC = -(1/2) C * dA / A
    dC_drho = -0.5 * C * dA_drho / A
    dC_dz   = -0.5 * C * dA_dz   / A

    # Helpful inverse radii
    inv_rho     = 1.0 / (rho + EPSILON_STABILITY)
    inv_rho2    = inv_rho * inv_rho

    # T = -K + (S * E / Delta)
    T = -K + (S * E) / Delta
    dT_drho = -dK_drho + ((dS_drho * E + S * dE_drho) * Delta - (S * E) * dD_drho) / (Delta * Delta)
    dT_dz   = -dK_dz   + ((dS_dz   * E + S * dE_dz  ) * Delta - (S * E) * dD_dz  ) / (Delta * Delta)

    # U = K + (S2 * E / Delta)
    U = K + (S2 * E) / Delta
    dU_drho = dK_drho + ((dS2_drho * E + S2 * dE_drho) * Delta - (S2 * E) * dD_drho) / (Delta * Delta)
    dU_dz   = dK_dz   + ((dS2_dz   * E + S2 * dE_dz  ) * Delta - (S2 * E) * dD_dz  ) / (Delta * Delta)

    # Field components in cylindrical:
    # B_rho = C * (z / rho) * T
    # B_z   = C * U
    B_rho = C * (z * inv_rho) * T
    B_z   = C * U

    # Partials in cylindrical
    # dB_rho/drho = dC/drho*(z/rho)*T + C*(z/rho)*dT/drho + C*z*T * d(1/rho)/drho
    #              = dC_drho*(z*inv_rho)*T + C*(z*inv_rho)*dT_drho - C*z*T*inv_rho2
    dBrho_drho = dC_drho * (z * inv_rho) * T + C * (z * inv_rho) * dT_drho - C * z * T * inv_rho2
    # dB_rho/dz   = dC/dz*(z/rho)*T + C*(1/rho)*T + C*(z/rho)*dT/dz
    dBrho_dz   = dC_dz   * (z * inv_rho) * T + C * inv_rho * T + C * (z * inv_rho) * dT_dz

    # dB_z/drho = dC/drho * U + C * dU/drho
    dBz_drho = dC_drho * U + C * dU_drho
    # dB_z/dz   = dC/dz   * U + C * dU/dz
    dBz_dz   = dC_dz   * U + C * dU_dz

    # Convert cylindrical partials to Cartesian Jacobian.
    # Bx = B_rho cosφ; By = B_rho sinφ; Bz = B_z
    # Using: ∂ρ/∂x = cosφ, ∂ρ/∂y = sinφ, ∂φ/∂x = -sinφ/ρ, ∂φ/∂y = cosφ/ρ
    # Results (compact, stable with inv_rho):
    Jxx = (cosphi * cosphi) * dBrho_drho + (B_rho * (sinphi * sinphi)) * inv_rho
    Jxy = (sinphi * cosphi) * (dBrho_drho - B_rho * inv_rho)
    Jxz = cosphi * dBrho_dz

    Jyx = Jxy
    Jyy = (sinphi * sinphi) * dBrho_drho + (B_rho * (cosphi * cosphi)) * inv_rho
    Jyz = sinphi * dBrho_dz

    Jzx = cosphi * dBz_drho
    Jzy = sinphi * dBz_drho
    Jzz = dBz_dz

    J = torch.stack([
        torch.stack([Jxx, Jxy, Jxz], dim=-1),
        torch.stack([Jyx, Jyy, Jyz], dim=-1),
        torch.stack([Jzx, Jzy, Jzz], dim=-1),
    ], dim=-2)

    return J
