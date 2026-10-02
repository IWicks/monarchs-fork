"""
Isabelle Wicks, Northumbria University (2/10/2026)

Functions to generate a non-uniform vertical grid used for near-surface processes.
"""

import numpy as np

def _nonuniform_depth_core(flat_depth, n_points, surface_depth, surface_fraction):
   
    """
    Core implementation, operating on a flat 1D array of depths.
    Returns shape (flat_depth.size, n_points).

    Columns with firn_depth <= surface_depth get a single uniform zone
    across their full depth. Columns deeper than surface_depth get a fine
    uniform zone from 0 to surface_depth, followed by a coarsening zone
    """
    
    n_surface = max(int(np.round(surface_fraction * n_points)), 2)
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



def generate_nonuniform_depth_scalar(firn_depth, n_points, surface_depth=1.0, surface_fraction=0.3):
    
    """
    Non-uniform depth coordinates for a single column.

    Parameters
    ----------
    firn_depth : float
        Total depth of the column [m].
    n_points : int
        Number of vertical grid points.
    surface_depth : float
        Depth of the fine near-surface zone [m].
    surface_fraction : float
        Fraction of n_points allocated to the near-surface zone.

    Returns
    -------
    z : ndarray, shape (n_points,)
    
    """
    flat_depth = np.array([float(firn_depth)])
    z = _nonuniform_depth_core(flat_depth, n_points, surface_depth, surface_fraction)
    
    return z[0]



def generate_nonuniform_depth_array(firn_depth, n_points, surface_depth=1.0, surface_fraction=0.3):
    
    """
    Non-uniform depth coordinates for an array of columns.

    Parameters
    ----------
    firn_depth : ndarray
        Total depth of the columns [m].
    n_points : int
        Number of vertical grid points.
    surface_depth : float
        Depth of the fine near-surface zone [m].
    surface_fraction : float
        Fraction of n_points allocated to the near-surface zone.

    Returns
    -------
    z : ndarray, shape firn_depth.shape + (n_points,)
    
    """
    firn_depth = np.asarray(firn_depth, dtype=float)
    orig_shape = firn_depth.shape
    flat_depth = firn_depth.ravel()

    z_flat = _nonuniform_depth_core(flat_depth, n_points, surface_depth, surface_fraction)

    return z_flat.reshape(orig_shape + (n_points,))



def generate_nonuniform_depth(firn_depth, n_points, surface_depth=1.0, surface_fraction=0.3):
    
    """
    Dispatches to the scalar or array implementation based on input type.
    """
    
    if np.isscalar(firn_depth):
        return generate_nonuniform_depth_scalar(firn_depth, n_points, surface_depth, surface_fraction)
    
    return generate_nonuniform_depth_array(firn_depth, n_points, surface_depth, surface_fraction)
   
