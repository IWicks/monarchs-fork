"""
Isabelle Wicks, Northumbria University (30/09/2026)

Functions to calculate the extinction coefficient of the vertical column and
shortwave penetration beneath the surface.
"""

import numpy as np

def extinction_coefficient(cell, beta_ice=2.5, beta_sfc=17.1, beta_water=0.0025, rho_sfc=500):
    
    """
    Calculates per-layer broadband extinction coefficient of the cell based on the solid and
    liquid water fraction. Interpolates between snow and ice extinction coefficient values from
    Bintanja and van den Broeke (1995). The Surface Energy Balance of Antarctic Snow and Blue Ice.
    
    rho_sfc anchors the 'snow-like' end of the interpolation to the same surface density used
    at initialisation. Change if initial surface density is different to 500 kg m^-3.
    
    Parameters
    -----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    beta_ice : float
        The extinction coefficient of ice [m^-1].
    beta_sfc : float
        The extinction coefficient of the surface snow/firn layer [m^-1].
    beta_water : float
        The extinction coefficient of water [m^-1].
    rho_sfc : float
        The prescribed surface density [kg m^-3].
    
    Returns
    -------
    beta_bulk : float
        The bulk extinction coeffiecient [m^-1].
    """
    
    if cell["blue_ice"] in (1,2):
        beta_matrix = np.full_like(cell["rho"], beta_ice)
        
    else:
        frac = np.clip(
            (cell["rho"] - rho_sfc) / (cell["rho_ice"] - rho_sfc), 0, 1
        )
        beta_matrix = beta_sfc * (1 - frac) + beta_ice * frac
    
    beta_bulk = (1 - cell["Lfrac"]) * beta_matrix + cell["Lfrac"] * beta_water
    
    return beta_bulk



def sw_penetration(cell, SW_in, alpha, dz):
    
    """
    Calculates shortwave penetration into the subsurface of the firn column, using Beer's law.
    
    Parameters
    -----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    SW_in : float
        Incoming shortwave (solar) radiation [W m^-2].
    alpha : float
        Effective surface albedo for shortwave radiation.
    dz : ndarray, dimension(cell.vert_grid)
        Depth of each vertical cell in the column.

    Returns
    -------
    SW_abs : ndarray, dimension(cell.vert_grid)
        The shortwave radiation over the vertical column [W m^-3].
    
    """
    
    # Obtain the bulk extinction coefficient for the vertical column
    beta_bulk = extinction_coefficient(cell)
    
    # Apply Beer's law to vertical column, calculating optical depth and transmitted SW
    tau_layer = beta_bulk * dz
    tau_top = np.cumsum(tau_layer) - tau_layer
    tau_bottom = np.cumsum(tau_layer)
    
    I0_net = (1 - alpha) * SW_in
    flux_top = I0_net * np.exp(-tau_top)
    flux_bottom = I0_net * np.exp(-tau_bottom)
    
    SW_abs = (flux_top - flux_bottom) / dz
    
    return SW_abs
