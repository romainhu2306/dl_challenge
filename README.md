# DEEP LEARNING CHALLENGE : Romain HÛ, Mohammed LBAKALI
This repository contains all the files related to our work, including data, python scripts and predictions.

## DATA
The data file includes :
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
- **predict.py**, containing blocks of code to make predictions for submission on codabench, saved in the predictions folder.

## SAVED MODELS
Contains the saved architectures of our trained models.
Because of its experimental nature, the competitive aggregation was not saved as its main interest does not lie in its performance.

## PREDICTIONS
Prediction files for submission on codabench with the test set.
Predictions were realized with temperature and pressure, and without using Covid markers.

## MISCELLANEOUS
- **X_test**, a torch tensor containing the preprocessed test set ;
- **y_scaler**, the scikit-learn StandardScaler that was used to scale electricity consumption.
