"""
Isabelle Wicks, Northumbria University (09/07/2026)

Functions to import, set up, and format blue_ice CSV into matrix that MONARCHS can use.
"""

import os
import numpy as np

def load_blue_ice(model_setup):
    """
    Load in the blue ice CSV and convert to a NumPy array, perform logic checks.
    
    Parameters
    ----------
    model_setup : .py file
        MONARCHS model_setup.py file used to define the model grid.
    
    Returns
    -------
    blue_ice : np.array
        A NumPy array of blue ice values.
        
    Raises
    ------
    IOError
        If the supplied file is not readable.
        
    ValueError
        If the grid is the wrong shape or if any values in the grid are not between 0 and 1.
    """
    
    # Checking for the existence of the file and that it can be read.
    if os.access(model_setup.blue_ice_input_filepath, os.R_OK):
        blue_ice = np.loadtxt(model_setup.blue_ice_input_filepath, dtype=np.float64, delimiter=',')
    else:
        raise IOError(f"The file {model_setup.blue_ice_input_filepath} is not readable.")
    
    # Checking that the RVf grid is the same shape as the model grid.
    if blue_ice.shape != (model_setup.col_amount, model_setup.row_amount):
        raise ValueError("The blue ice grid is the wrong shape!")
    
    # Checking that all values in the grid are between 0 and 1.
    if np.any((blue_ice<0) | (blue_ice>1)):
        raise ValueError("All grid values must be between 0 and 1!")
        
    return blue_ice
