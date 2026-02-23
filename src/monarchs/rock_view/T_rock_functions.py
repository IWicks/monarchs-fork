"""
Isabelle Wicks, Northumbria University (23/02/2026)

Functions to import, set up, and format T_rock NetCDF4 dataset into dictionary that MONARCHS can use.
"""

import netCDF4
import os.path

def T_rock_to_vars(T_rock_input, met_timestep, total_days, start_index=0, chunk_size=366):
    
    """
    Take in an input T_rock NetCDF file, and convert it into a dictionary that can be read in by MONARCHS.
    Modified from monarchs-ice/monarchs/src/met_data/import_ERA5.py.

    Parameters
    ----------
    T_rock_input : str
        Path to a NetCDF file of rock surface temperature input.

    Returns
    -------
    rock_dict : dict
        Dictionary of gridded output, with variable names and formatting suitable for loading into MONARCHS.
    """
    # TODO (Izzy) - make sure datetime64 data is imported and indexed correctly
    
    rock_dict = {}
    
    # Determine indices for start and end of the year.
    if total_days * met_timestep > start_index + (met_timestep * chunk_size):
       end_index = start_index + (met_timestep * chunk_size)
    else:
       end_index = start_index + (met_timestep * total_days - start_index)
       
    # Chunk size = 366 as data contains 29/02 values.

    start_index = int(start_index)
    end_index = int(end_index)
    T_rock_data = netCDF4.Dataset(T_rock_input)
    
    if end_index > len(T_rock_data.variables["valid_time"]):
        raise ValueError(
            f"monarchs.rock_view.T_rock_to_variables: End index {end_index} is greater than the length of "
            f"the data available ({len(T_rock_data.variables['time'])} timesteps) in the input NetCDF file."
            f" Please check your input data is large enough, or adjust your chosen number of days to compensate'."
        )
        
    try:
        rock_dict["time"] = T_rock_data.variables["time"][start_index:end_index]
    except KeyError:
        try:
            rock_dict["time"] = T_rock_data.variables["valid_time"][start_index:end_index]
        except:
            raise KeyError(
                'Time variable "time" or "valid_time" not found in the input T_rock NetCDF. Check your input data, or amend <monarchs.rock_view.T_rock_to_variables> to use the key that is in your data.'
            )
            
    try:
        rock_dict["temperature"] = T_rock_data.variables["tcsl"][start_index:end_index]
    except:
        raise KeyError('Temperature variable "tcsl" or "temp" not found in the input T_rock NetCDF. Check your input data, or amend <monarchs.rock_view.T_rock_to_variables> to use the key that is in your data.'
            )
    
    T_rock_data.close()
    
    return rock_dict

def setup_T_rock(model_setup):
    """
    Read in and set up a rock temperature NetCDF dataset, only if an RVf grid is provided.
    Modified from monarchs-ice/monarchs/src/core/setup_met_data.py.

    Parameters
    ----------
    model_setup : str
        MONARCHS model_setup.py file.

    Returns
    -------
    T_rock_vars : dict
        Dictionary of gridded output, with variable names and formatting suitable for loading into MONARCHS.
    """
    
    if os.path.isfile(model_setup.RVf_input_filepath) == True: # Only perform this function if an RVf grid is provided
        
        match model_setup.t_rock_timestep:
            case 'hourly':
                index = 1
            case 'three_hourly':
                index = 3
            case 'daily':
                index = 24
            case int():
                index = model_setup.model_setup.t_rock_timestep
            case _:
                raise ValueError(
                    'monarchs.rock_view.setup_t_rock: t_rock_timestep should be an integer, "hourly", "three_hourly" or "daily". See documentation for model_setup.t_rock_timestep for details.'
                ) # Easier to do as if..else statement or match case?
            
        chunk_size = 100
        model_years = max(1, model_setup.num_days // chunk_size + 1)
        if model_setup.num_days % chunk_size == 0:
            model_years -= 1  # If the number of days is a multiple of chunk_size (default 365), we don't need an extra year.
        for year in range(model_years):
            start_index = year * chunk_size * 24 / index
            timesteps_per_day = 24 / index
    
            total_days = model_setup.num_days
            T_rock_vars = T_rock_to_vars(model_setup.t_rock_input_filepath, timesteps_per_day, total_days, start_index=start_index,
            chunk_size=chunk_size)
            
        return T_rock_vars
    # TODO (Izzy) - check indentation of return
