"""
Isabelle Wicks, Northumbria University (02/03/2026)

Function to load T_rock CSV dataset into timeseries that MONARCHS can use.
"""

import os
import pandas as pd

def load_T_rock(model_setup):
     
     """
     Load a rock temperature CSV dataset, only if an RVf grid is provided.

     Parameters
     ----------
     model_setup : str
         MONARCHS model_setup.py file.

     Returns
     -------
     T_rock : float
         Timeseries of T_rock data.
         
     Raises
     ------
     IOError
         If the supplied file is not readable.
     """
     
     # Checking for the existence of the file and that it can be read.
     if os.access(model_setup.T_rock_input_filepath, os.R_OK):
         T_rock_data = pd.read_csv(model_setup.T_rock_input_filepath, delimiter=',', header='infer',
                                   parse_dates=['valid_time'], date_format='%Y-%m-%mT%hh:%mm:%ss') # Change column header and datetime format if required.
     else:
         raise IOError(f"The file {model_setup.T_rock_input_filepath} is not readable.")
    
     return T_rock_data
