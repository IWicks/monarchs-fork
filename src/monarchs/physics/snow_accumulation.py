"""
Created on Tue May  9 17:04:04 2023

@author: sglsb6
"""

import numpy as np
from monarchs.core import utils
from monarchs.core.nonuniform_grid import compute_box_edges, generate_nonuniform_depth_scalar
from monarchs.physics.grid_remap import nonuniform_remap


def snowfall(cell, snow_depth, snow_rho, snow_T):
    """
    Adds snowfall to surface of model and regrids model to incorporate it. This snow is added to the firn or lake
    depth, depending on the current state of the cell.
    Called in <timestep_loop>.
    TODO - what happens if we have a lid? Gets added to top and we just regrid as normal.

    For the firn case, fresh snow is incorporated via nonuniform_remap, treating it as one additional 'virtual
    box' at the top of the old column, appended to old grid properties (Sfrac/Lfrac/firn_temperature) before
    remapping onto a newly generated (non-(uniform) grid for the increased firn depth.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    snow_depth : float
        Depth of the snow, read in from the meteorological data input [m]
    snow_rho : float
        Snow density, either read in from the meteorological data input, or assumed 300 [kg m^-3]
    snow_T : float
        Snow temperature, either read in from meterological data input, or assumed 260 [K]

    Returns
    -------
    None.
    """
    if snow_depth >= 1.2e-5:

        if cell["blue_ice"] in (1, 2):
            cell["blue_ice"] = 2
            cell["blue_ice_transition_day"] = cell["day"]  # Reset albedo decay clock
        
        if cell["lake"]:
            cell["lake_depth"] += snow_depth * snow_rho / cell["rho_water"]
            
        else:
            original_mass = utils.calc_mass_sum(cell)
            
            if snow_T > 273.15:
                snow_T = 273.149
                
            old_firn_depth = cell["firn_depth"] + 0
            old_vertical_profile = cell["vertical_profile"].copy()
            old_edges = compute_box_edges(old_vertical_profile)
            
            cell["firn_depth"] += snow_depth
            new_firn_depth = cell["firn_depth"]
            
            # Regenerate non-uniform point positions for the new, taller grid
            new_vertical_profile = generate_nonuniform_depth_scalar(new_firn_depth, cell["vert_grid"])
            new_edges = compute_box_edges(new_vertical_profile)

            # Anchoring grid reference points to the fixed base of the column
            old_edges_b = old_firn_depth - old_edges[::-1]
            new_edges_b = new_firn_depth - new_edges[::-1]
            
            Sfrac_b = cell["Sfrac"][::-1]
            Lfrac_b = cell["Lfrac"][::-1]
            firn_T_b = cell["firn_temperature"][::-1]
            
            # Extend the old column data with one virtual box spanning [old_firn_depth, new_firn_depth]
            ext_old_edges_b = np.concatenate([old_edges_b, [new_firn_depth]])
            ext_Sfrac_b = np.concatenate([Sfrac_b, [snow_rho / cell["rho_ice"]]])
            ext_Lfrac_b = np.concatenate([Lfrac_b, [0.0]])
            ext_firn_T_b = np.concatenate([firn_T_b, [snow_T]])
            
            Sfrac_new = nonuniform_remap(ext_old_edges_b, ext_Sfrac_b, new_edges_b)[::-1]
            Lfrac_new = nonuniform_remap(ext_old_edges_b, ext_Lfrac_b, new_edges_b)[::-1]
            firn_T_new = nonuniform_remap(ext_old_edges_b, ext_firn_T_b, new_edges_b)[::-1]

            
            if np.any(cell["Sfrac"] < -0.001):
                print(np.where(cell["Sfrac"] < -0.001))
                raise ValueError("Sfrac before snow > 0")
            if np.any(cell["Sfrac"] > 1.001):
                print(np.where(cell["Sfrac"] > 1.001))
                raise ValueError("Sfrac before snow > 1")
            
            if np.any(Sfrac_new < -0.001):
                print(np.where(Sfrac_new < -0.001))
                raise ValueError("Sfrac after snow < 0")
            if np.any(Sfrac_new > 1.001):
                print(np.where(Sfrac_new > 1.001))
                raise ValueError("Sfrac after snow > 1")
            if np.any(Lfrac_new < -0.001):
                print(np.where(Lfrac_new < -0.001))
                raise ValueError("Lfrac after snow < 0")
                
            cell["Sfrac"] = Sfrac_new
            cell["Lfrac"] = Lfrac_new
            cell["firn_temperature"] = firn_T_new
            cell["vertical_profile"] = new_vertical_profile
            
            assert (
                abs(utils.calc_mass_sum(cell) - (original_mass + snow_depth * snow_rho * (1 - cell["RVf"])))
                < 1.5 * 10**-7
            )


def densification(cell, t_steps_per_day):
    """
    TODO - not important now, but implement for later.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.

    t_steps_per_day : int

    Returns
    -------

    """
    dt = 1 / (8 * t_steps_per_day * 365)
    g = 9.81
    ec = 60
    eg = 42.4
    R = 8.3144598
    T_av = 264.56010894609415
    e = np.exp(-ec / (R * cell["firn_temperature"][0]) + eg / (R * T_av))
    cell["rho"] = cell["Sfrac"] * cell["rho_ice"] + cell["Lfrac"] * cell["rho_water"]
    b = 0.4953 * (1000 / 350)
    for i in range(cell["vert_grid"]):
        if cell["rho"][i] > cell["rho_ice"]:
            cell["rho"][i] = cell["rho_ice"]
        if cell["rho"][i] < 550:
            c = 0.07
        else:
            c = 0.03
        d_rho = c * b * g * (cell["rho_ice"] - cell["rho"][i]) * e * dt
        cell["rho"][i] = cell["rho"][i] + d_rho
        cell["Sfrac"][i] = cell["rho"][i] / cell["rho_ice"]
        if cell["Sfrac"][i] > 1:
            print("Snow accumulation has caused Sfrac to be > 1")
