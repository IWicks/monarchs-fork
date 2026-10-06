"""
Isabelle Wicks, Northumbria University (6/10/2026)

Functions to handle meltwater processes for a non-uniform vertical grid.

Generalises the original regrid_after_melt() 'scale' adjustment to handle any number of
old boxes being partially or fully consumed by melt. Preserves the same physics as the
original function: liquid water in a partially-melted vertical box is retained/concentrated
into its surviving portion, and liquid in a fully-melted box has no surviving firn pore space
to stay in, so it is converted to meltwater, as is melted ice.

"""

import numpy as np

def lost_overlap(old_edges_b, firn_depth_new, firn_depth_old):
    
    """
    Calculates how much each old vertical box overlaps the melted region.
 
    All inputs must already be in base-anchored coordinates (distance from the fixed base,
    not depth from the moving surface).
 
    Parameters
    ----------
    old_edges_b : ndarray, shape (n_old + 1,)
        Old box boundaries, base-anchored.
    firn_depth_new : float
        New (post-melt) total column depth [m].
    firn_depth_old : float
        Old (pre-melt) total column depth [m].
 
    Returns
    -------
    lost_overlap_b : ndarray, shape (n_old,)
        Overlap length of each old box with the melted-away region [firn_depth_new,
        firn_depth_old] [m].
    """
    
    lost_lo, lost_hi = firn_depth_new, firn_depth_old
    overlap_lo = np.maximum(lost_lo, old_edges_b[:-1])
    overlap_hi = np.minimum(lost_hi, old_edges_b[1:])
    
    return np.clip(overlap_hi - overlap_lo, 0, None)



def melt_retain_and_loss(Sfrac_b, Lfrac_b, old_box_widths_b, lost_overlap_b, rho_ice, rho_water, tol=1e-9):
    
    """
    Adjusts Lfrac to retain liquid water in partially-melted old boxes and calculates the total
    generated meltwater.
 
    Melted ice is removed via conversion to meltwater, whereas liquid water stays within the firn
    wherever any pore space remains to store it.
 
    Parameters
    ----------
    Sfrac_b, Lfrac_b : ndarray, shape (n_old,)
        Old solid/liquid fractions, base-anchored order (i.e. reversed from the usual surface-indexed
        storage order).
    old_box_widths_b : ndarray, shape (n_old,)
        Old box widths (np.diff of old_edges_b).
    lost_overlap_b : ndarray, shape (n_old,)
        Output of lost_overlap.
    rho_ice, rho_water : float
        Densities, for converting melted ice volume to water-equivalent volume [kg m^-3].
    tol : float
        Numerical tolerance for classifying a box as "fully" vs "partially" melted.
 
    Returns
    -------
    Lfrac_b_adj : ndarray, shape (n_old,)
        Lfrac with partially-melted boxes' liquid concentrated into the remaining pore space.
    meltwater : float
        Total generated meltwater: melted ice from every box impacted by regridding, plus the full
        liquid content of any entirely-melted vertical box [m.w.e.].
    """
    
    full_melt = lost_overlap_b >= (old_box_widths_b - tol)
    part_melt = (lost_overlap_b > tol) & (~full_melt)
 
    Lfrac_b_adj = Lfrac_b.copy()
    Lfrac_b_adj[part_melt] = (Lfrac_b[part_melt] * old_box_widths_b[part_melt]
                              / (old_box_widths_b[part_melt] - lost_overlap_b[part_melt]))
 
    # NB: retain/concetrate rescaling above can push Lfrac_b_adj past the physical pore space
    # limit (Sfrac - 1). This is not capped here, but is instead resolved by the mechanisms
    # in percolation_functions.cal_saturation. See regrid_after_melt() for how meltflag gets 
    # set so percolation() picks up the affected levels.

    
    meltwater_liquid_lost = np.sum(Lfrac_b[full_melt] * old_box_widths_b[full_melt])
    lost_Sfrac_integral = np.sum(lost_overlap_b * Sfrac_b)
    meltwater = lost_Sfrac_integral * (rho_ice / rho_water) + meltwater_liquid_lost
 
    return Lfrac_b_adj, meltwater
