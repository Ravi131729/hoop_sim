import numpy as np

def hoop_contact_point(
    hoop_center,
    hoop_radius,
    R_wb,
    n_g=np.array([0.0, 0.0, 1.0]),
    eps=1e-8
):
    """
    Compute closest point on a hoop to a ground plane.

    Parameters
    ----------
    hoop_center : (3,) array
        Hoop center in world frame
    hoop_radius : float
        Hoop radius
    R_wb : (3,3) array
        Rotation matrix from body to world
    n_g : (3,) array
        Ground normal (unit), default z-up
    eps : float
        Tolerance for degeneracy

    Returns
    -------
    p_contact : (3,) array
        Closest point on hoop to ground
    """

    # --- hoop plane normal in world (body y-axis) ---
    n_h = R_wb @ np.array([0.0, 1.0, 0.0])

    # --- fix normal to point away from ground ---
    if np.dot(n_h, n_g) < 0.0:
        n_h = -n_h

    # --- project ground normal into hoop plane ---
    n_parallel = n_g - np.dot(n_g, n_h) * n_h
    norm_np = np.linalg.norm(n_parallel)

    # --- degenerate case: hoop plane parallel to ground ---
    if norm_np < eps:
        # choose any  in-plane direction (body x-axis)
        d = R_wb @ np.array([1.0, 0.0, 0.0])
        d = d / np.linalg.norm(d)
    else:
        # direction toward ground inside hoop plane
        d = -n_parallel / norm_np

    # --- contact point ---
    p_contact = hoop_center + hoop_radius * d

    return p_contact
