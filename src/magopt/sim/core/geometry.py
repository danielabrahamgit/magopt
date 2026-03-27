import torch


def build_segments(wire_pts: torch.Tensor, closed: bool):
    """
    Build segment endpoints from wire points.

    Args:
    -----
    wire_pts : torch.Tensor
        Wire points with shape [N, 3].
    closed : bool
        Whether the wire is closed.

    Returns:
    --------
    tuple[torch.Tensor, torch.Tensor, int]
        Segment starts, segment ends, and segment count.
    """
    pts = wire_pts
    N = pts.shape[0]
    if closed:
        return pts, torch.roll(pts, shifts=-1, dims=0), N
    return pts[:-1], pts[1:], N - 1


def build_adjacency(lim: int, closed: bool, device: torch.device) -> torch.Tensor:
    """
    Build adjacency mask for segment pairs.

    Args:
    -----
    lim : int
        Number of segments.
    closed : bool
        Whether first and last segments are adjacent.
    device : torch.device
        Device for output tensor.

    Returns:
    --------
    torch.Tensor
        Boolean matrix [lim, lim] marking adjacent segment pairs.
    """
    idx = torch.arange(lim, device=device)
    dist_ij = (idx[:, None] - idx[None, :]).abs()
    is_adjacent = dist_ij == 1
    if closed:
        is_adjacent |= dist_ij == (lim - 1)
    return is_adjacent
