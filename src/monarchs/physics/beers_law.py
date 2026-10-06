"""
Isabelle Wicks, Northumbria University (6/10/2026)

Functions to calculate the extinction coefficient of the vertical column and shortwave penetration
beneath the surface for all surface types.
"""

import numpy as np

def extinction_coefficient(cell, tau_ice=2.5, tau_sfc=17.1, tau_water=0.0025, rho_sfc=500):
    
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
    tau_ice : float
        The extinction coefficient of ice [m^-1].
    tau_sfc : float
        The extinction coefficient of the surface snow/firn layer [m^-1].
    tau_water : float
        The extinction coefficient of water [m^-1].
    rho_sfc : float
        The prescribed surface density [kg m^-3].
    
    Returns
    -------
    tau_bulk : ndarray, dimension(cell.vert_grid)
        The bulk extinction coeffiecient [m^-1].
    """

    if cell["blue_ice"] in (1,2):
        tau_matrix = np.full_like(cell["rho"], tau_ice)
        
    else:
        frac = np.clip(
            (cell["rho"] - rho_sfc) / (cell["rho_ice"] - rho_sfc), 0, 1
        )
        tau_matrix = tau_sfc * (1 - frac) + tau_ice * frac
    
    tau_bulk = (1 - cell["Lfrac"]) * tau_matrix + cell["Lfrac"] * tau_water
    
    return tau_bulk



def sw_penetration(cell, SW_in, alpha, box_dz):
    
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
    dz_box : float
        Box thickness surrounding each vertical grid point, from 
        box_dz.compute_box_dz(cell["vertical_profile"]) [m].


    Returns
    -------
    SW_abs : ndarray, dimension(cell.vert_grid)
        The shortwave radiation over the vertical column [W m^-3].
    """
    
    # Obtain the bulk extinction coefficient for the vertical column
    tau_bulk = extinction_coefficient(cell)
    
    # Apply Beer's law to vertical column, calculating optical depth and transmitted SW
    od_layer = tau_bulk * box_dz
    od_top = np.cumsum(od_layer) - od_layer
    od_bottom = np.cumsum(od_layer)
    
    I0_net = (1 - alpha) * SW_in
    flux_top = I0_net * np.exp(-od_top)
    flux_bottom = I0_net * np.exp(-od_bottom)
    
    SW_abs = (flux_top - flux_bottom) / box_dz
    
    return SW_abs



def extinction_coefficient_lid(cell, tau_lid_ice=2.5):
    
    """
    Calculates per-layer broadband extinction coefficient of the true lid.
    
    rho_lid is fixed at rho_ice (917 kg m^-3), treating the lid as fully consolidated,
    bubble-free ice throughout, meaning it is optically closer to blue ice (Bintanja and
    van den Broeke, 1995, tau_ice) thus is handled with the same extinction coefficient.
    
    
    Parameters
    -----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    tau_lid_ice : float
        The extinction coefficient of ice [m^-1].
    
    Returns
    -------
    tau_lid : ndarray, dimension(cell.vert_grid_lid)
        The per-layer extinction coeffiecient [m^-1].
    """

    tau_lid = np.full_like(cell["rho_lid"], tau_lid_ice)
    
    return tau_lid



def sw_penetration_lid(cell, SW_in, alpha, box_dz):
    
    """
    Calculates shortwave penetration into the subsurface of the true lid, using Beer's law.
    
    Parameters
    -----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    SW_in : float
        Incoming shortwave (solar) radiation [W m^-2].
    alpha : float
        Effective surface albedo for shortwave radiation.
    dz_box : float
        Box thickness surrounding each vertical grid point, from 
        box_dz.compute_box_dz(cell["vertical_profile"]) [m].

    Returns
    -------
    SW_abs_lid : ndarray, dimension(cell.vert_grid)
        The shortwave radiation over the vertical column [W m^-3].
    """
    
    # Obtain the bulk extinction coefficient for the vertical column
    tau_lid = extinction_coefficient_lid(cell)
    
    # Apply Beer's law to vertical column, calculating optical depth and transmitted SW
    od_layer = tau_lid * box_dz
    od_top = np.cumsum(od_layer) - od_layer
    od_bottom = np.cumsum(od_layer)
    
    I0_net = (1 - alpha) * SW_in
    flux_top = I0_net * np.exp(-od_top)
    flux_bottom = I0_net * np.exp(-od_bottom)
    
    SW_abs_lid = (flux_top - flux_bottom) / box_dz
    
    return SW_abs_lid
