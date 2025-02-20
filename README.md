# Deep learning challenge : Romain HÛ, Mohammed LBAKALI
This repository contains :
- The data files, including train and test sets along with annual CO2 emission for France in .csv format, and meteorological data in parquet format ;
- Python scripts including :
    - A utils.py script, containing functions for data processing, training and validation loops, plotting residuals ;
    - A models.py script, containing the Pytorch code for each model ;
    - A train.py script, containing the code for :
        - Data processing : each block can be ran to add the variables you want to add ;
        - Training :
            - The baseline model ;
            - The overparameterized model ;
            - The overparameterized sine model ;
            - The orthogonal aggregation model ;
            - The baseline and sine aggregated model ;
            - The competitive aggregation model.
        - Plotting the validation error and residuals of each model.
