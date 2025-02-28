import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import holidays
import utils
import models
import joblib

X_test = torch.load("X_test")
y_scaler = joblib.load("y_scaler.pkl")

################################################################################
# BASELINE MODEL
################################################################################

model = torch.load("saved_models/base_model.pt", weights_only = False)
pred = utils.simple_predict(X_test, model, y_scaler)
pred.to_csv("predictions/base_prediction.csv")


################################################################################
# OVERSINE MODEL
################################################################################

model = torch.load("saved_models/oversine_model.pt", weights_only = False)
pred = utils.simple_predict(X_test, model, y_scaler)
pred.to_csv("predictions/oversine_prediction.csv")


################################################################################
# OVERBASE MODEL
################################################################################

model = torch.load("saved_models/overbase_model.pt", weights_only = False)
pred = utils.simple_predict(X_test, model, y_scaler)
pred.to_csv("predictions/overbase_prediction.csv")


################################################################################
# ORTHOGONAL AGGREGATED MODEL
################################################################################

mod1 = torch.load("saved_models/ortho_mod1.pt", weights_only = False)
mod2 = torch.load("saved_models/ortho_mod2.pt", weights_only = False)
mod3 = torch.load("saved_models/ortho_mod3.pt", weights_only = False)
pred = utils.aggreg_predict(X_test, mod1, mod2, mod3, y_scaler)
pred.to_csv("predictions/ortho_prediction.csv")


################################################################################
# OVERPARAMETERIZED BASELINE AND SINE AGGREAGATED MODEL
################################################################################

mod1 = torch.load("saved_models/aggreg_mod1.pt", weights_only = False)
mod2 = torch.load("saved_models/aggreg_mod2.pt", weights_only = False)
mod3 = torch.load("saved_models/aggreg_mod3.pt", weights_only = False)
pred = utils.aggreg_predict(X_test, mod1, mod2, mod3, y_scaler)
pred.to_csv("predictions/aggreg_prediction.csv")