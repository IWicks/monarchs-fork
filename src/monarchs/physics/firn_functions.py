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
from monarchs.core.nonuniform_grid import compute_box_dz, compute_box_edges, generate_nonuniform_depth_scalar
from monarchs.physics.beers_law import sw_penetration

MAX_PHYS_DHDT = 1.0 # m per timestep, deliberately extreme value to catch unphysical melt events

def firn_column(
    cell,
    dt,
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

    # Compute variables needed for non-uniform grid calculations
    alpha = surface_fluxes.sfc_albedo(cell["melt"], cell["exposed_water"], cell["lid"], cell["lake"],
                                      cell["lake_depth"], cell["blue_ice"], cell["day"], cell["blue_ice_transition_day"])
    dz_box = compute_box_dz(cell["vertical_profile"]) # Compute box thicknesses surrounding each point in the vertical grid
    old_edges = compute_box_edges(cell["vertical_profile"])
    old_firn_depth = cell["firn_depth"]
    height_change_surface = 0.0
    SW_abs = sw_penetration(cell, SW_in, alpha, dz_box)
    
    args = [cell, dt, LW_in, SW_in, T_air, p_air, T_dp, T_rock, wind, SW_abs, dz_box]
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
        height_change_surface = calc_height_change(
            cell, dt, LW_in, SW_in, T_air, p_air, T_dp, T_rock, wind, root[0], SW_abs, dz_box
        )

        if prescribed_height_change is not False:
            height_change_surface = 0.05
        if np.isnan(height_change_surface):
            raise ValueError("Height change is NaN")
            
        # Recompute values as melting has occurred
        alpha = surface_fluxes.sfc_albedo(cell["melt"], cell["exposed_water"], cell["lid"], cell["lake"],
                                      cell["lake_depth"], cell["blue_ice"], cell["day"], cell["blue_ice_transition_day"])
        SW_abs = sw_penetration(cell, SW_in, alpha, dz_box)
        args = cell, dt, LW_in, SW_in, T_air, p_air, T_dp, T_rock, wind, SW_abs, dz_box
        root, fvec, success_fixedsfc, info = solver.firn_heateqn_solver(
            x, args, fixed_sfc=True, solver_method=heateqn_solver
        )

        if success_fixedsfc:
            cell["firn_temperature"] = root

    elif success:
        cell["firn_temperature"] = root
        cell["melt"] = False
    else:
        pass

    # Determining column compaction due to internal melting
    compaction = np.zeros(cell["vert_grid"]) # Initialise array of zeroes for internal compaction
    for v_lev in range (1, cell["vert_grid"]): # Ignoring the zero-thickness surface skin
        if cell["firn_temperature"][v_lev] > 273.15: # Scan for internal cells that are at the melt temperature
            Sfrac_before = cell["Sfrac"][v_lev]    
            cell["meltflag"][v_lev] = 1
            calc_interior_melt(cell, v_lev, dz_box) # Calculate the melting and adjust Sfrac/Lfrac/T in that cell
            Sfrac_after = cell["Sfrac"][v_lev]
            # Fractional change of Sfrac, multiplied by the cell's width
            compaction[v_lev] = ((Sfrac_before - Sfrac_after) / Sfrac_before) * dz_box[v_lev]
    melt_indices = np.nonzero(compaction)[0] # Find the locations where internal melting occurred
    melt_widths = compaction[melt_indices]
    melt_positions = old_firn_depth - old_edges[melt_indices] # Find the base-anchored position of the melt events
    
    if height_change_surface > 0 or len(melt_widths) > 0:
        regrid_combined_melt(cell, melt_positions, melt_widths, height_change_surface)

    if percolation_toggle:
        percolation_functions.percolation(cell, dt, perc_time_toggle=perc_time_toggle)

    cell["rho"] = cell["Sfrac"] * cell["rho_ice"] + cell["Lfrac"] * cell["rho_water"]
    new_mass = utils.calc_mass_sum(cell)

    assert abs(original_mass - new_mass) < 1.5 * 10**-7
    return root0

def regrid_after_melt(cell, height_change, lake=False):
    """
    After melting occurs at the surface, subtract the amount of melting from the firn height, convert it into meltwater,
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
    
    if height_change > MAX_PHYS_DHDT:
        raise ValueError(
            f"Height change during melt ({height_change:.4f} m) "
            f"exceeds physically reasonable bounds"
            )
    
    original_mass = utils.calc_mass_sum(cell)
    old_firn_depth = cell["firn_depth"] + 0

    tol = 1e-9
    if height_change >= old_firn_depth - tol:
        raise ValueError(
            f"Height change ({height_change}) >= "
            f"firn depth ({old_firn_depth}) - whole-column melt, "
            f"x={cell['column']}, y={cell['row']}"
            )
    
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
    old_box_widths_b = np.diff(old_edges_b)

    Sfrac_b = cell["Sfrac"][::-1]
    Lfrac_b = cell["Lfrac"][::-1]
    firn_T_b = cell["firn_temperature"][::-1]

    # Calculate how many old/new boxes overlap
    lost_overlap_b = lost_overlap(old_edges_b, new_firn_depth, old_firn_depth)
    Lfrac_b_adj, meltwater = melt_retain_and_loss(Sfrac_b, Lfrac_b, old_box_widths_b,
        lost_overlap_b, cell["rho_ice"], cell["rho_water"])

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


def calc_height_change(cell, timestep, LW_in, SW_in, T_air, p_air, T_dp, T_rock, wind, surf_T,
                      SW_abs, dz_box):
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
    SW_abs : ndarray, dimension(cell.vert_grid)
        The absorbed shortwave radiation across the vertical column [W m^-3].
    dz_box : ndarray, dimension(cell.vert_grid)
        Box thickness surrounding each vertical grid point, from nonuniform_grid.compute_box_dz(cell["vertical_profile"]) [m].

    Returns
    -------
    dHdt: float
        The height change over the vertical column [m].
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
        cell["size_dx"],
        cell["blue_ice"],
        cell["day"],
        cell["blue_ice_transition_day"],
    )

    Q -= (1 - alpha) * SW_in  # Subtracting surface SW absorption, will be accounted for across column via Beer's law
    Q += SW_abs[0] * dz_box[0] # Converting volumetric term to flux term (W/m^3 to W/m^2), adding the fraction of incoming energy
                              # that applies to the surface cell, accounting for the zero-thickness skin.
    
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
    elif dHdt > MAX_PHYS_DHDT:
        raise ValueError(
            f"Height change during melt ({dHdt:.4f} m) exceeds physically reasonable bounds"
            )
    elif np.isnan(dHdt):
        print(f"NaN height change diagnostic — column={cell['column']}, row={cell['row']}")
        print(f"  Q={Q}, T[0]={cell['firn_temperature'][0]}, T[1]={cell['firn_temperature'][1]}")
        print(f"  k_sfc={k_sfc}, Sfrac[0]={cell['Sfrac'][0]}")
        print(f"  SW_abs[0]={SW_abs[0]}, dz_box[0]={dz_box[0]}")
        print(f"  vertical_profile[:5]={cell['vertical_profile'][:5]}")
        print(f"  firn_depth={cell['firn_depth']}")
        raise ValueError("Height change during melt is NaN")
        """
        pass
        print("...")
        raise ValueError("Height change during melt is NaN")
        """
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


def calc_interior_melt(cell, v_lev, dz_box):
    
    """
    The mirror of percolation_functions.calc_refreezing. Calculates melting of water within the
    firn column at a specified layer v_lev due to SW penetration. This melting changes Sfrac into
    Lfrac, and adds any excess energy into any remaining liquid water to heat it up.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    v_lev : int
        Vertical level at which to calculate the refreezing of the firn.
    dz_box : ndarray, shape (vert_grid,)
        Box thickness surrounding each point [m].
        
    Raises
    ------
    ValueError
        If Sfrac is less than 0 or Lfrac is greater than 1, raise a ValueError as the model
        has entered an unphysical state.
        
    Returns
    -------
    None (amends cell inplace)
    """
    
    T_excess = cell["firn_temperature"][v_lev] - 273.15
    cp = 7.16 * cell["firn_temperature"][v_lev] + 138
    T_melt_all = cell["L_ice"] / cp
    
    # If the cell doesn't have enough thermal mass to absorb all of the excess heat, and the whole cell melts
    if T_excess >= T_melt_all:
        melt_vol = (
            cell["Sfrac"][v_lev]
            * cell["rho_ice"]
            * dz_box[v_lev]
            / cell["rho_water"]
        )
        cell["Lfrac"][v_lev] += melt_vol / dz_box[v_lev]
        cell["daily_melt"] += melt_vol
        if cell["Lfrac"][v_lev] > 1:
            raise ValueError("Lfrac > 1 in interior melt calculation")
        cell["Sfrac"][v_lev] = 0
        # Heating now applies to water, so use the specific heat capacity of water instead
        cell["firn_temperature"][v_lev] = 273.15 + (cp * T_excess - cell["L_ice"]) / cell["cp_water"] 
    # If the cell has enough thermal mass to absorb the excess heat, and only some of the cell melts
    else:
        melt_vol = (
            cell["Sfrac"][v_lev]
            * cp
            * dz_box[v_lev]
            * T_excess
            / cell["L_ice"]
        )
        cell["Sfrac"][v_lev] = cell["Sfrac"][v_lev] - (melt_vol / dz_box[v_lev])
        if cell["Sfrac"][v_lev] < 0:
            raise ValueError("Sfrac < 0 in interior melt calculation")
        cell["Lfrac"][v_lev] = cell["Lfrac"][v_lev] + (
            melt_vol / dz_box[v_lev] * (cell["rho_ice"] / cell["rho_water"])
        )
        cell["daily_melt"] += melt_vol * (cell["rho_ice"] / cell["rho_water"])
        cell["firn_temperature"][v_lev] = 273.15


def cumulative_loss(query_points, melt_positions, melt_widths):
    
    """
    For every upper box edge, computes the cumulative width lost at or below the edge's position. Used to track
    the height change due to internal melting and firn compaction.

    Parameters
    ----------
    query_points : ndarray, shape (m,)
        The base-anchored positions at which to evaluate the cumulative loss function, independent of the number
        of melt events, typically every edge of the old grid.
    melt_positions : ndarray, shape (n_events,)
        The base-anchored positions where the melt events occurred.
    melt_widths : ndarray, shape (n_events,)
        The widths removed by each melt event. melt_positions[i] and melt_widths[i] refer to the same event, with
        the two arrays being the same length and correspondingly ordered [m].
        
    Returns
    -------
    padded_total_width : ndarray, shape (m,)
        The height change at a given point due to compaction at or beneath that point [m]. 
    """
    
    # Sorting the melt events by depth, then ordering both the positions and widths by this indexing
    sort = np.argsort(melt_positions) 
    sorted_positions = melt_positions[sort]
    sorted_widths = melt_widths[sort]
    
    # Finding the running total lost width, padded with 0 to represent "zero melt events yet"
    padded_total_width = np.concatenate(([0.0], np.cumsum(sorted_widths)))
    
    # Finding the number of melt events at or below query_points[i]
    counts = np.searchsorted(sorted_positions, query_points, side='right')
    
    return padded_total_width[counts]


def regrid_combined_melt(cell, melt_positions, melt_widths, height_change_surface, lake=False):
    
    """
    After melting occurs in the column interior, find the cumulative shift of positions within the column due to
    melting beneath their location, then compact these boxes, retaining their solid fraction and concentrating their
    liquid fraction (water either present in the cell beforehand, or produced by internal melting in this timestep,
    via calc_interior_melt). Then add the surface melting contribution to this height change, and interpolate the
    entire column to the new (non-uniform) vertical profile accounting for this cumulative height change.

    Parameters
    ----------
    cell : numpy structured array
        Element of the model grid we are operating on.
    melt_positions : ndarray, shape (n_events,)
        The base-anchored positions where the melt events occurred.
    melt_widths : ndarray, shape (n_events,)
        The widths removed by each melt event. melt_positions[i] and melt_widths[i] refer to the same event, with
        the two arrays being the same length and correspondingly ordered [m].
    height_change_surface : float
        The height change at the surface due to surface melting [m].
    lake : bool, optional
        Flag to determine whether a lake is present or not. This is contained here so that we can re-use the bulk
        of this algorithm, but with some small changes to reflect the different situation that occurs when a lake is
        present.
    
    Returns
    -------
    None

    """
    
    original_mass = utils.calc_mass_sum(cell)
    bs = cell["Sfrac"].copy() # Retained for Sfrac>1 diagnostic message
    old_firn_depth = cell["firn_depth"] + 0
    old_vertical_profile = cell["vertical_profile"].copy()
    old_edges = compute_box_edges(old_vertical_profile)
    tol = 1e-9
    
    # Anchoring grid reference points to the fixed base of the column
    old_edges_b = old_firn_depth - old_edges[::-1]
    Sfrac_b = cell["Sfrac"][::-1]
    Lfrac_b = cell["Lfrac"][::-1]
    firn_T_b = cell["firn_temperature"][::-1]
    
    old_box_widths_b = np.diff(old_edges_b)
    
    # Finding the total shift in all positions in the column due to compaction
    cumulative_shift = cumulative_loss(old_edges_b, melt_positions, melt_widths)
    
    # Find the base-anchored edge positions after the total shift, and the new firn height after all melting
    old_edges_b_shift = old_edges_b - cumulative_shift
    new_firn_depth = old_edges_b_shift[-1] - height_change_surface
    
    # Regenerate non-uniform point positions for the new, shorter grid
    new_vertical_profile = generate_nonuniform_depth_scalar(new_firn_depth, cell["vert_grid"])
    new_edges = compute_box_edges(new_vertical_profile)
    new_edges_b = new_firn_depth - new_edges[::-1]
    new_box_widths = np.diff(new_edges)
    
    if np.isnan(cell["firn_temperature"]).any():
        raise ValueError("NaN in firn temperature before regridding")

    # Calculate the width lost of all boxes, then adjust the interior cells' Sfrac and Lfrac based on this loss width,
    # retaining and concentrating any meltwater into the column
    box_losses = np.diff(cumulative_shift)
    effective_box_widths_b = old_box_widths_b - box_losses
    Sfrac_b_adj = Sfrac_b * (old_box_widths_b / (old_box_widths_b - box_losses))
    Lfrac_b_adj_compaction = Lfrac_b * (old_box_widths_b / (old_box_widths_b - box_losses))
    
    # Find the lost region from the surface in base-anchored coordinates
    lost_overlap_b_surface = lost_overlap(old_edges_b_shift, new_firn_depth, old_edges_b_shift[-1])
    
    # Find the resulting meltwater and adjusted Lfrac at the surface
    Lfrac_b_adj, meltwater = melt_retain_and_loss(Sfrac_b_adj, Lfrac_b_adj_compaction, effective_box_widths_b,
        lost_overlap_b_surface, cell["rho_ice"], cell["rho_water"])

    # Returning to surface-anchored grid reference points
    Sfrac_hold = nonuniform_remap(old_edges_b_shift, Sfrac_b_adj, new_edges_b)[::-1]
    Lfrac_hold = nonuniform_remap(old_edges_b_shift, Lfrac_b_adj, new_edges_b)[::-1]
    firn_T_hold = nonuniform_remap(old_edges_b_shift, firn_T_b, new_edges_b)[::-1]
    
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
        print("total height change = ", np.sum(melt_widths))
        print("melt positions = ", melt_positions)
        print("melt widths = ", melt_widths)
        print("firn depth = ", cell["firn_depth"])
        raise ValueError("Sfrac > 1 in firn regridding")
    
    # Flag oversaturation to be treated by percolation_functions.percolation, like regrid_after_melt
    oversaturated = (cell["Sfrac"] + cell["Lfrac"]) > (1 + tol)
    cell["meltflag"][oversaturated] = 1
    
    cell["vertical_profile"] = new_vertical_profile
    cell["firn_depth"] = new_firn_depth

    assert abs(utils.calc_mass_sum(cell) - original_mass) < 1.5 * 10**-7
    