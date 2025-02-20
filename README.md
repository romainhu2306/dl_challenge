# DEEP LEARNING CHALLENGE : Romain HÛ, Mohammed LBAKALI
This repository contains all the files related to our work, including data, python scripts and predictions.

## DATA
The data files include :
- The train and test sets from codabench in .csv format ;
- Annual CO2 emission for France in .csv format
- Meteorological data from codabench in parquet format.

## PYTHON SCRIPTS
This repository contains 3 python scripts :
- **utils.py**, containing functions for data processing, training and validating, plotting residuals and custom loss functions ;
- **models.py**, containing the Pytorch code for each model :
    - Baseline model ;
    - Overparameterized model ;
    - Overparameterized sine model ;
    - Orthogonal aggregation model ;
    - Baseline and sine aggregation model ;
    - Competitive aggregation model ;
- **train.py**, containing blocks of code to process the data, train the models and display their results.

## PREDICTIONS
Prediction file for submission on codabench with the test set.
