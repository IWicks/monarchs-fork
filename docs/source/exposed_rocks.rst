Including exposed rocks in the model run
========================================
Why include exposed rocks?
--------------------------
The default installation of MONARCHS assumes a uniform snowy ice shelf surface. The exposed rock release accounts for nunatuks
and other rock features that may impact the surface energy balance and meltwater production, especially in winter. This MONARCHS
release can be used by supplying data in ``model_setup.py``, which will activate the exposed rock functions within the model.

Required files
--------------------------
The ``RVf_input_filepath`` variable requires a 'rock view fraction' grid in ``.csv`` format. This grid has the same resolution as your
model domain (i.e., same number of rows and columns), with each grid cell containing a value between 0-1 to represent the fraction of
rock in that area. An example RVf grid has been provided in the ``data`` folder of the MONARCHS repository. Specify the path to the
``.csv`` file in the ``RVf_input_filepath`` variable in ``model_setup.py``.

If using an RVf grid, it is best to update the value of ``epsilon_rock`` in the ``surface_fluxes.py`` file of your local MONARCHS copy
to represent the geology of your study area. An example epsilon value is provided in the file, but this may not be the best value for all
study areas.

The ``T_rock_input_filepath`` variable requires a dataset of rock surface temperatures in ``.csv`` format. This file must match the length
and resolution of your model run, i.e., at hourly intervals for the study period. MONARCHS currently does not support looping over a year-long
file for the duration of the model run. Only one temperature is required for the entire model domain at each timestep, as MONARCHS does not
currently support spatially varying rock surface temperatures. Specify the path to the ``.csv`` file in the ``T_rock_input_filepath`` variable
in ``model_setup.py``.

No rock inclusion
--------------------------
If you wish to run MONARCHS without exposed rocks, simply comment out the variables in ``model_setup.py``. The model should then calculate the
surface energy balance and lateral movement processes using the default installation and uniform surface assumptions.
