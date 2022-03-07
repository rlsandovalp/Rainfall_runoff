Once the raw data is saved in a given folder, the procedure to organize the data is the following:

1) Run the convert_X.py script to fill NaNs
    If you want to convert flowrate data, take into consideration that the equation of the station must be modified
2) Run the join_series.py script to unify all the data associated with a single variable and station in a unique file.


Comput_ET.py:    File to compute the potential evapotranspiration based on the Romanenko equation.