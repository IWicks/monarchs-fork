"""
Isabelle Wicks, Northumbria University (7/10/2026)

General remapping between two 1D non-uniform grids, to be used in regridding functions.

Generalises the old uniform-grid-only "weight_1/weight_2" two-neighbor overlap scheme to
handle a new box overlapping any number of old boxes.
"""

import numpy as np

def nonuniform_remap(old_edges, q_old, new_edges, tol=1e-9):
    
    """
    Remaps the box height-change calculation for a non-uniform grid.
    
    For each new box j, computes how much j overlaps every old box (not just one or two
    neighbours), using the minimum of the uppers minus the maximum of the lowers (clipped
    at the domain edges (vertical_profile[0] and vertical_profile[-1])), then takes a
    weighted average, normalised by the new box's width.
    
    'Width' is used to refer to the vertical width of boxes.

    Parameters
    ----------
    old_edges : ndarray, shape (n_old + 1,)
        Box boundary positions for the old grid.
    q_old : ndarray, shape (n_old,)
        Box-averaged quantity on the old grid.
    new_edges : ndarray, shape (n_new + 1,)
        Box boundary positions for the new grid.
    tol : float
        Tolerance for the monotonicity check.

    Returns
    -------
    q_new : ndarray, shape (n_new,)
        Box-averaged quantity remapped onto the new grid.
        
    Rasies
    ------
    ValueError
        If old_edges or new_edges are not monotonically increasing.
    """
    
    if np.any(np.diff(old_edges) < -tol):
        raise ValueError("nonuniform_remap: old_edges is not monotonically increasing.")
    if np.any(np.diff(new_edges) < -tol):
        raise ValueError("nonuniform_remap: new_edges is not monotonically increasing.")
    
    n_new = len(new_edges) - 1
    q_new = np.zeros(n_new)

    for j in range(n_new):
        L_j, R_j = new_edges[j], new_edges[j + 1]
        new_box_length = R_j - L_j

        # Overlap with every old box - a large negative value is not itself an error, but means
        # this particular old box doesn't overlap this particular new box at all, which is expected
        # for most box pairs
        overlap_left = np.maximum(L_j, old_edges[:-1])
        overlap_right = np.minimum(R_j, old_edges[1:])
        overlap_length = np.clip(overlap_right - overlap_left, 0, None)

        q_new[j] = np.sum(overlap_length * q_old) / new_box_length

    return q_new
