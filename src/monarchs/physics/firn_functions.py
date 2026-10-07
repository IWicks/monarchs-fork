"""
Module containing functions relating to the firn column. Some physics is contained in percolation.py.
"""

import numpy as np
from monarchs.physics import percolation_functions
from monarchs.physics import surface_fluxes
from monarchs.physics import solver
from monarchs.physics.grid_remap import nonuniform_remap
from monarchs.physics.nonuniform_firn_funcs import lost_overlap, melt_retain_and_loss
from monarchs.core import utils
from monarchs.core.nonuniform_grid import compute_box_edges, generate_nonuniform_depth_scalar

def firn_column(
    cell,
    dt,
    dz,
    LW_in,
    SW_in,
    T_air,
    p_air,
    T_dp,
    T_rock,
    wind,
    toggle_dict,
    prescribed_height_change=False,
):
    """
    Perform the various processes applied to the firn each timestep where we don't have exposed water at the surface.

    The logic works as follows:
    Solve heat equation for firn

    If surface temperature is above melting point of water
    (i,e melting takes place):

    - Determine the height change as a result of the melting
    - Regrid everything to take account for this deformation
    - Re-solve heat equation, now with fixed surface temperature
    - Calculate the amount of water added to the surface as a result of melt
    - Percolate that water down, taking into account any lids or lakes that may have formed.

    Otherwise:
    - Update cell temperature and continue.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    dt : int
        Number of seconds in the current timestep [s]
    dz : float
        Height of each vertical point in the cell. [m]
    LW_in : float
        Downwelling longwave radiation at the current timestep. [W m^-2]
    SW_in : float
        Downwelling shortwave radiation at the current timestep. [W m^-2]
    T_air : float
        Surface air temperature at the current timestep. [K]
    p_air : float
        Surface air pressure at the current timestep. [Pa]
    T_dp : float
        Dewpoint temperature of the air at the surface at the current timestep. [K]
    T_rock : float
        Surface temperature of the rock at the current timestep. [K]
    wind : float
        Wind speed at the surface at the current timestep. [m s^-1]
    toggle_dict : dict
        Dictionary containing some switches that affect the running of the model.

    prescribed_height_change : float, optional
        For testing purposes, it can be useful to set a prescribed height change to force the firn to lose height
        regardless of the meteorological conditions, with the corresponding increase in water. [m]

    Returns
    -------
    None (amends cell inplace)
    """
    original_mass = utils.calc_mass_sum(cell)
    percolation_toggle = toggle_dict["percolation_toggle"]
    perc_time_toggle = toggle_dict["perc_time_toggle"]
    heateqn_solver = 'hybr'
    x = cell["firn_temperature"]
    #x = np.clip(x, 0, 273.15)
    args = [cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, T_rock, wind]
    root, fvec, success, info = solver.firn_heateqn_solver(
        x, args, fixed_sfc=False, solver_method=heateqn_solver
    )
    # print(f'Root[0] = {root[0]}')
    root0 = root[0]
    if root[0] > 273.15:
        cell["meltflag"][0] = 1
        cell["melt"] = True
        if cell["blue_ice"] == 2:
            cell["blue_ice"] = 1
            cell["blue_ice_transition_day"] = np.nan
        height_change = calc_height_change(
            cell, dt, LW_in, SW_in, T_air, p_air, T_dp, T_rock, wind, root[0]
        )

        if prescribed_height_change is not False:
            height_change = 0.05
        if np.isnan(height_change):
            raise ValueError("Height change is NaN")

        dz = cell["firn_depth"] / cell["vert_grid"]
        args = cell, dt, dz, LW_in, SW_in, T_air, p_air, T_dp, T_rock, wind
        root, fvec, success_fixedsfc, info = solver.firn_heateqn_solver(
            x, args, fixed_sfc=True, solver_method=heateqn_solver
        )

        if success_fixedsfc:
            cell["firn_temperature"] = root
        regrid_after_melt(cell, height_change)

    elif success:
        cell["firn_temperature"] = root
        cell["melt"] = False
    else:
        pass

    if percolation_toggle:
        percolation_functions.percolation(cell, dt, perc_time_toggle=perc_time_toggle)

    cell["rho"] = cell["Sfrac"] * cell["rho_ice"] + cell["Lfrac"] * cell["rho_water"]
    new_mass = utils.calc_mass_sum(cell)

    assert abs(original_mass - new_mass) < 1.5 * 10**-7
    return root0

def regrid_after_melt(cell, height_change, lake=False):
    """
    After melting occurs, subtract the amount of melting from the firn height, convert it into meltwater,
    and interpolate the entire column to the new (non-uniform) vertical profile accounting for this height change.
    
    Liquid water retention physics are preserved and generalised: a partially-consumed old box has its liquid
    concentrated into the remaining pore space, and a fully-consumed old box's liquid becomes meltwater, the same
    as melted ice. Excess meltwater is either converted into surface liquid water fraction, or if there is a lake,
    into lake height.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    height_change : float
        Change in the firn height as a result of melting. [m]
    lake : bool, optional
        Flag to determine whether a lake is present or not. This is contained here so that we can re-use the bulk
        of this algorithm, but with some small changes to reflect the different situation that occurs when a lake is
        present.

    Returns
    -------
    None
    """
    original_mass = utils.calc_mass_sum(cell)
    old_firn_depth = cell["firn_depth"] + 0

    tol = 1e-9
    if height_change >= old_firn_depth - tol:
        raise ValueError(
            f"regrid_after_melt: height change ({height_change}) >= "
            f"firn depth ({old_firn_depth}) - whole-column melt, "
            f"x={cell["column"]}, y={cell["row"]}"
    
    old_vertical_profile = cell["vertical_profile"].copy()
    old_edges = compute_box_edges(old_vertical_profile)
    
    bs = cell["Sfrac"].copy() # Retained for Sfrac>1 diagnostic message

    cell["firn_depth"] -= height_change
    new_firn_depth = cell["firn_depth"]

    # Regenerate non-uniform point positions for the new, shorter grid
    new_vertical_profile = generate_nonuniform_depth_scalar(new_firn_depth, cell["vert_grid"])
    new_edges = compute_box_edges(new_vertical_profile)
    new_box_widths = np.diff(new_edges) # 'Widths' here refers to vertical width of boxes

    if np.isnan(cell["firn_temperature"]).any():
        raise ValueError("NaN in firn temperature before regridding")

    # Anchoring grid reference points to the fixed base of the column
    old_edges_b = old_firn_depth - old_edges[::-1]
    new_edges_b = new_firn_depth - new_edges[::-1]
    old_box_widths_b = np.diff(old_edges)

    Sfrac_b = cell["Sfrac"][::-1]
    Lfrac_b = cell["Lfrac"][::-1]
    firn_T_b = cell["firn_temperature"][::-1]

    # Calculate how many old/new boxes overlap
    lost_overlap_b = lost_overlap(old_edges_b, new_firn_depth, old_firn_depth)
    Lfrac_b_adj, meltwater = melt_retain_and_loss(Sfrac_b, Lfrac_b, old_box_widths_b,
        lost_overlap_b, cell["rho_ice"], cell["rho_water"],)

    # Returning to surface-anchored grid reference points
    Sfrac_hold = nonuniform_remap(old_edges_b, Sfrac_b, new_edges_b)[::-1]
    Lfrac_hold = nonuniform_remap(old_edges_b, Lfrac_b_adj, new_edges_b)[::-1]
    firn_T_hold = nonuniform_remap(old_edges_b, firn_T_b, new_edges_b)[::-1]
    
    cell["Sfrac"] = Sfrac_hold
    cell["Lfrac"] = Lfrac_hold
    cell["firn_temperature"] = firn_T_hold

    if np.isnan(firn_T_hold).any():
        print(firn_T_hold)
        print(cell["firn_temperature"])
        print(cell["column"])
        print(cell["row"])
        raise ValueError("NaN in firn temperature after regridding")

    cell["daily_melt"] += meltwater
    cell["Lfrac"][0] += meltwater / new_box_widths[0] # Convert to fraction here
    
    if lake:
        if cell["Lfrac"][0] + cell["Sfrac"][0] > 1:
            excess_water = cell["Lfrac"][0] + cell["Sfrac"][0] - 1
            cell["lake_depth"] += excess_water * new_box_widths[0]
            cell["Lfrac"][0] = 1 - cell["Sfrac"][0]
        assert abs(utils.calc_mass_sum(cell) - original_mass) < 1.5 * 10**-7

    if np.isnan(cell["firn_temperature"]).any():
        print(cell["firn_temperature"])
        raise ValueError("NaN in firn temperature after regridding")

    if np.any(cell["Sfrac"] > 1.00000000001):
        where = np.where(cell["Sfrac"] > 1)
        print(where[0])
        print("Old Sfrac = ", bs[where])
        print("Sfrac = ", cell["Sfrac"][where])
        print("x = ", cell["column"], "y = ", cell["row"])
        print("height change = ", height_change)
        print("firn depth = ", cell["firn_depth"])
        raise ValueError("Sfrac > 1 in firn regridding")

    # NB: the retain/concentrate rescaling can leave Sfrac+Lfrac > 1 at any level, not just the surface,
    # since melting can span many fine near-surface boxes. Rather than being capped here, this is
    # flagged for percolation_functions.percolation to resolve via the existing "fill column upwards
    # from impermeable layer" mechanism.
    
    oversaturated = (cell["Sfrac"] + cell["Lfrac"]) > (1 + tol)
    cell["meltflag"][oversaturated] = 1
    
    cell["vertical_profile"] = new_vertical_profile

    assert abs(utils.calc_mass_sum(cell) - original_mass) < 1.5 * 10**-7


def calc_height_change(cell, timestep, LW_in, SW_in, T_air, p_air, T_dp, T_rock, wind, surf_T):
    """
    Determine the amount of firn height change that arises due to melting.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    timestep : float
        Number of seconds in each timestep. [s]
    LW_in : float
        Downwelling longwave radiation at the current timestep. [W m^-2]
    SW_in : float
        Downwelling shortwave radiation at the current timestep. [W m^-2]
    T_air : float
        Surface air temperature at the current timestep. [K]
    p_air : float
        Surface air pressure at the current timestep. [Pa]
    T_dp : float
        Dewpoint temperature of the air at the surface at the current timestep. [K]
    T_rock : float
        Surface temperature of the rock at the current timestep. [K]
    wind : float
        Wind speed at the surface at the current timestep. [m s^-1]
    surf_T : float
        Calculated surface temperature of the firn from the initial (non-fixed surface) implementation of the heat
        equation. [K]

    Returns
    -------

    """
    epsilon_ice = 0.98
    sigma = 5.670374e-8
    dz = np.diff(cell["vertical_profile"])[0]
    L_fus = 334000
    if cell["firn_temperature"][0] > 273.14999999 and cell["firn_temperature"][0] < 273.151:
        cell["firn_temperature"][0] = 273.15
    if cell["firn_temperature"][1] > 273.14999999 and cell["firn_temperature"][1] < 273.151:
        cell["firn_temperature"][1] = 273.15
    k_sfc = (
        1000 * 2.24 * 10**-3
        + 5.975
        * 10**-6
        * (273.15 - (cell["firn_temperature"][0] + cell["firn_temperature"][1]) / 2)
        ** 1.156
    )
    Q, Flat, Fsens, alpha = surface_fluxes.sfc_flux(
        cell["melt"],
        cell["exposed_water"],
        cell["lid"],
        cell["lake"],
        cell["lake_depth"],
        LW_in,
        SW_in,
        T_air,
        p_air,
        T_dp,
        T_rock,
        wind,
        surf_T,
        cell['RVf'],
        cell["blue_ice"],
        cell["day"],
        cell["blue_ice_transition_day"],
    )
    dHdt = (
        timestep
        * (
            Q
            - epsilon_ice * sigma * cell["firn_temperature"][0] ** 4
            - k_sfc * ((cell["firn_temperature"][0] - cell["firn_temperature"][1]) / dz)
        )
        / (cell["rho_ice"] * (cell["Sfrac"][0] * L_fus))
    )

    if 0 > dHdt > -0.01:
        dHdt = 0
    elif dHdt < -0.01:
        raise ValueError(
            "Height change during melt is negative, and outside the bounds of a numerical error"
        )
    elif np.isnan(dHdt):
        pass
        print("...")
        raise ValueError("Height change during melt is NaN")
    return dHdt


def interp_nb(x_vals, x, y):
    """
    Wrapper function for np.interp. This function exists purely so that an alternative interpolation algorithm can be
    used throughout the code without needing to change every instance. Named interp_nb as this function also has
    Numba compatibility in its default form.

    Parameters
    ----------
    x_vals : array_like
        New coordinates that we want to interpolate our input y values onto.
    x : array_like
        Original coordinates of our y values.
    y : array_like
        Values we want to interpolate.

    Returns
    -------
    res : array_like
        values from y, interpolated onto our new grid of x_vals
    """
    res = np.interp(x_vals, x, y)
    return res
