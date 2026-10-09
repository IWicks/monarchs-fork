"""
Isabelle Wicks, Northumbria University (9/10/2026)

Functions to generate a non-uniform vertical grid and vertical boxes used for near-surface
processes.

Default surface_depth=0.25 and surface_fraction=0.075 are tuned against the worst-
case extinction coefficient in the model (beta_snow=17.1 m^-1, e-fold depth ~0.0585 m)
for 400 vertical points. This covers ~4.3 e-folds (98.6% of absorbed shortwave) at ~7
grid cells per e-fold.

If beta_snow or the number of vertical points changes, surface_depth and surface_fraction
may need to be re-tuned for your specific model setup.
"""

import numpy as np

def compute_box_dz(vertical_profile):
    
    """
    Computes the box thickness surrounding each point in vertical_profile.
 
    Each grid point represents a node. The box for each node extends halfway to each
    neighbouring point. The top and bottom boxes are capped at the domain edges
    (vertical_profile[0] and vertical_profile[-1]).
 
    Parameters
    ----------
    vertical_profile : ndarray, shape (vert_grid,)
        Depth coordinate of each grid point [m].
 
    Returns
    -------
    dz_box : ndarray, shape (vert_grid,)
        Box thickness surrounding each point [m].
    """
    
    midpoints = (vertical_profile[:-1] + vertical_profile[1:]) / 2
    box_boundaries = np.concatenate(([vertical_profile[0]], midpoints, [vertical_profile[-1]]))
    dz_box = np.diff(box_boundaries)
    
    return dz_box



def compute_box_edges(vertical_profile):
    
    """
    Computes the box boundary positions surrounding each point in vertical_profile.
    
    Each grid point represents a node, where its box extends halfway to each neighbouring
    point, capped at the domain edges (vertical_profile[0] and vertical_profile[-1]) for
    the top/bottom points.

    Parameters
    ----------
    vertical_profile : ndarray, shape (vert_grid,)
        Depth coordinate of each grid point [m].

    Returns
    -------
    box_edges : ndarray, shape (vert_grid + 1,)
        Box boundary positions, suitable for use with conservative_remap.
    """
    
    midpoints = (vertical_profile[:-1] + vertical_profile[1:]) / 2
    box_edges = np.concatenate(([vertical_profile[0]], midpoints, [vertical_profile[-1]]))
    
    return box_edges



def _nonuniform_depth_core(flat_depth, n_points, surface_depth, n_surface):
   
    """
    Core implementation, operating on a flat 1D array of depths.
    Returns shape (flat_depth.size, n_points).

    Columns with firn_depth <= surface_depth get a single uniform zone
    across their full depth. Columns deeper than surface_depth get a fine
    uniform zone from 0 to surface_depth, followed by a coarsening zone
    """
    
    n_surface = max(int(n_surface), 2)
    n_deep = n_points - n_surface

    z = np.zeros((flat_depth.size, n_points))
    shallow = flat_depth <= surface_depth

    # Shallow columns: one uniform zone spanning the full depth
    if np.any(shallow):
        depths = flat_depth[shallow]
        frac_full = np.linspace(0.0, 1.0, n_points)
        z[shallow] = np.outer(depths, frac_full)

    # Normal columns: fine near-surface zone and coarsening deep zone
    if np.any(~shallow):
        depths = flat_depth[~shallow]

        frac1 = np.linspace(0.0, 1.0, n_surface)
        zone1 = np.outer(np.ones_like(depths), surface_depth * frac1)

        remaining = depths - surface_depth
        # Drop 0 to avoid repeating last point of zone1
        frac2 = np.linspace(0.0, 1.0, n_deep + 1)[1:]
        zone2 = surface_depth + np.outer(remaining, frac2)

        z[~shallow] = np.concatenate([zone1, zone2], axis=-1)

    return z



def generate_nonuniform_depth_scalar(firn_depth, n_points, surface_depth=0.25, n_surface=30):
    
    """
    Non-uniform depth coordinates for a single column. surface_depth requirea tuning based on number
    of vertical grid points used in model run (current tuning based on default value of 400).

    Parameters
    ----------
    firn_depth : float
        Total depth of the column [m].
    n_points : int
        Number of vertical grid points.
    surface_depth : float
        Depth of the fine near-surface zone [m].
    n_surface : int
        Number of points allocated to the near-surface zone (fixed value, not a fraction
        of n_points).

    Returns
    -------
    z : ndarray, shape (n_points,)
    """
    
    flat_depth = np.array([float(firn_depth)])
    z = _nonuniform_depth_core(flat_depth, n_points, surface_depth, n_surface)
    
    return z[0]



def generate_nonuniform_depth_array(firn_depth, n_points, surface_depth=0.25, n_surface=30):
    
    """
    Non-uniform depth coordinates for an array of columns. surface_depth requirea tuning based on number
    of vertical grid points used in model run (current tuning based on default value of 400).

    Parameters
    ----------
    firn_depth : ndarray
        Total depth of the columns [m].
    n_points : int
        Number of vertical grid points.
    surface_depth : float
        Depth of the fine near-surface zone [m].
    n_surface : int
        Number of points allocated to the near-surface zone (fixed value, not a fraction
        of n_points).

    Returns
    -------
    z : ndarray, shape firn_depth.shape + (n_points,)
    """
    
    firn_depth = np.asarray(firn_depth, dtype=float)
    orig_shape = firn_depth.shape
    flat_depth = firn_depth.ravel()

    z_flat = _nonuniform_depth_core(flat_depth, n_points, surface_depth, n_surface)

    return z_flat.reshape(orig_shape + (n_points,))



def generate_nonuniform_depth(firn_depth, n_points, surface_depth=0.25, n_surface=30):
    
    """
    Dispatches to the scalar or array implementation based on input type.
    """
    
    if np.isscalar(firn_depth):
        return generate_nonuniform_depth_scalar(firn_depth, n_points, surface_depth, n_surface)
    
    return generate_nonuniform_depth_array(firn_depth, n_points, surface_depth, n_surface)
