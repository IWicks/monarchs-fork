"""
Isabelle Wicks, Northumbria University (05/02/2026)

Load in the rock view fraction CSV to be used by the model grid.
"""

import numpy as np
import os
import os.path 

def load_RVf(model_setup, RVf_input_filepath):
    """
    Load in the rock view fraction (RVf) CSV and convert to a NumPy array, perform logic checks.
    
    Parameters
    ----------
    model_setup : .py file
        MONARCHS model_setup.py file used to define the model grid.
    RVf_input_filepath : str
        Filepath for the rock view fraction CSV.
        The file specified must be a .csv, with all values being between 0 and 1.
        The grid must be the same size as the model grid specified in model_setup.py. Currently only square grids are supported.
    
    Returns
    -------
    RVf : np.array
        A NumPy array of rock view fraction values.
        
    Raises
    ------
    IOError
        If the supplied file is not readable.
        
    ValueError
        If the grid is the wrong shape or if any values in the grid are not between 0 and 1.
    """
    
    # Checking for the existence of the file and that it can be read.
    if os.access(RVf_input_filepath, os.R_OK):
        RVf = np.loadtxt(RVf_input_filepath, dtype=np.float64, delimiter=',')
    else:
        raise IOError(f"The file {RVf_input_filepath} is not readable.")
    
    # Checking that the RVf grid is the same shape as the model grid.
    if RVf.shape != (model_setup.col_amount, model_setup.row_amount):
        raise ValueError("The RVf grid is the wrong shape!")
    
    # Checking that all values in the grid are between 0 and 1.
    if np.any((RVf<0) | (RVf>1)):
        raise ValueError("All grid values must be between 0 and 1!")
        
    return RVf
