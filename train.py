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


# Getting device.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Loading files.
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
meteo = pd.read_parquet('data/meteo.parquet')
co2 = pd.read_csv('data/co2_emission_france.csv')


# Converting to datetime.
train['date'] = pd.to_datetime(train['date'], utc = True)
test['date'] = pd.to_datetime(test['date'], utc = True)
meteo['date'] = pd.to_datetime(meteo['date'], utc = True)


# Filling nan values for metropolises.
train['day_of_year'] = train['date'].dt.day_of_year
train['hour'] = train['date'].dt.hour
'''
Each metropolis is grouped by day of year and hour, and the missing values are filled
with the mean of the group.
'''
for col in train.columns:
  if col not in ['date', 'day_of_year', 'hour']:
    train[col] = train[col].fillna(train.groupby(['day_of_year', 'hour'])[col].transform('mean'))

train.drop(columns = ['day_of_year', 'hour'], inplace = True)


# Adding CO2 emissions per year.
scaler = StandardScaler()
co2.drop(columns = 'Units', inplace = True)
co2['co2'] = scaler.fit_transform(co2[['co2']])

train['year'] = train['date'].dt.year
test['year'] = test['date'].dt.year

train = train.merge(co2, how = 'left', on = 'year')
test = test.merge(co2, how = 'left', on = 'year')

train.drop(columns = 'year', inplace = True)
test.drop(columns = 'year', inplace = True)


def add_meteo_var(var_name, suffix, train, test, meteo):
    '''
    Adds a given meteorological variable to the train set.
    The variables are pivoted by station, normalized and upsampled to half-hour frequency.
    '''
    scaler = StandardScaler()
    meteo[var_name] = scaler.fit_transform(meteo[[var_name]])

    # Pivoting feature by station.
    var = meteo[['date', var_name, 'numer_sta']]
    var = pd.pivot_table(var, values = var_name, index = 'date', columns = 'numer_sta')

    # Keeping only the stations with less than 30% of Nan values.
    var = var.loc[:, var.isna().mean() < .3]

    # Upsampling to half-hour frequency.
    var = var.resample('30min').interpolate(method = 'linear', limit_direction = 'both')

    # Dividing train and test sets.
    var_train = var.loc['2017-02-13 00:30:00+00:00':'2021-12-31 22:30:00+00:00'].reset_index()
    var_test = var.loc['2021-12-31 23:00:00+00:00':].reset_index()

    # Merging with train and test sets.
    train = train.merge(var_train, on = 'date', how = 'left', suffixes = ('', suffix))
    test = test.merge(var_test, on = 'date', how = 'left', suffixes = ('', suffix))

    return train, test


# Adding any meteorological variables we want.
# Feel free to choose which variables to add : 't', 'ff' and 'pres' are recommended.
train, test = utils.add_meteo_var('t', '_t', train, test, meteo)
train, test = utils.add_meteo_var('ff', '_ff', train, test, meteo)
train, test = utils.add_meteo_var('pres', '_pres', train, test, meteo)
train, test = utils.add_meteo_var('vv', '_vv', train, test, meteo)
train, test = utils.add_meteo_var('n', '_n', train, test, meteo)
train, test = utils.add_meteo_var('rr12', '_rr12', train, test, meteo)


# Filling a few remaining NaNs in the test set.
test.interpolate(method = 'linear', limit_direction = 'both', inplace = True)


# Extracting temporal features.
train = utils.extract_date(train)
test = utils.extract_date(test)


# Adding days-off.
fr_holidays = holidays.France()
train['BH'] = train['date'].apply(lambda x: x in fr_holidays).astype(int)
test['BH'] = test['date'].apply(lambda x: x in fr_holidays).astype(int)


# Adding covid confinement.
train['covid'] = (train['date'] >= '2020-03-17') & (train['date'] <= '2021-05-03')
test['covid'] = (test['date'] >= '2020-03-17') & (test['date'] <= '2021-05-03')
train['covid'] = train['covid'].astype(int)
test['covid'] = test['covid'].astype(int)


# Dropping original date column.
train.drop(columns = 'date', inplace = True)
test.drop(columns = 'date', inplace = True)


# Preparing datasets.
y_scaler = StandardScaler()
'''
The train set contains the first 55000 rows.
The validation set contains the last 15000 rows.
'''
X_train = torch.tensor(train.iloc[:55000, 25:].values, dtype = torch.float32)
y_train = torch.tensor(y_scaler.fit_transform(train.iloc[:55000, :25]), dtype = torch.float32)
X_valid = torch.tensor(train.iloc[65000:, 25:].values, dtype = torch.float32)
y_valid = torch.tensor(y_scaler.fit_transform(train.iloc[65000:, :25]), dtype = torch.float32)

X_test = torch.tensor(test.values, dtype = torch.float32)

train_set = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 1024, shuffle = True)


# Training base model.
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
base_model = models.baseline(input_dim, output_dim).to(device)
criterion = nn.MSELoss()
base_model = utils.simple_train(base_model, train_loader, criterion, .01, 50)

table, loss = utils.simple_valid(base_model, X_valid, criterion, y_scaler)
print(f"Validation loss: {loss:.4f}")

utils.plot_residuals(table, 0, y_valid)


# Training oversine model.
oversine_model = models.oversine(input_dim, output_dim).to(device)
oversine_model = utils.simple_train(oversine_model, train_loader, criterion, .01, 50)

table, loss = utils.simple_valid(oversine_model, X_valid, criterion, y_scaler)
print(f"Validation loss: {loss:.4f}")

utils.plot_residuals(table, 0, y_valid)


# Training overbase model.
overbase_model = models.overbase(input_dim, output_dim).to(device)
overbase_model = utils.simple_train(overbase_model, train_loader, criterion, .01, 50)

table, loss = utils.simple_valid(overbase_model, X_valid, criterion, y_scaler)
print(f"Validation loss: {loss:.4f}")

utils.plot_residuals(table, 0, y_valid)


# Training orthogonal aggregator.
mod1 = models.baseline(input_dim, output_dim).to(device)
mod2 = models.basesine(input_dim, output_dim).to(device)
mod3 = models.basesine(input_dim, output_dim).to(device)

mod1, mod2, mod3 = utils.aggreg_train(mod1, mod2, mod3, train_loader, criterion, .01, 200)

table, loss = utils.aggreg_valid(mod1, mod2, mod3, X_valid, criterion, y_scaler)
print(f"Validation loss: {loss:.4f}")

utils.plot_residuals(table, 0, y_valid)


# Training overbase and oversine aggregator.
mod1 = models.overbase(input_dim, output_dim).to(device)
mod2 = models.oversine(input_dim, output_dim).to(device)
mod3 = baseline(input_dim, output_dim).to(device)

mod1, mod2, mod3 = utils.aggreg_train(mod1, mod2, mod3, train_loader, criterion, .01, 200)

table, loss = utils.aggreg_valid(mod1, mod2, mod3, X_valid, criterion, y_scaler)
print(f"Validation loss: {loss:.4f}")

utils.plot_residuals(table, 0, y_valid)


# Training competitive aggregator.
mod1 = models.baseline(input_dim, output_dim).to(device)
mod2 = models.basesine(input_dim, output_dim).to(device)
mod3 = models.linear_aggreg(.5, .5).to(device)

mod1, mod2, mod3 = utils.competitive_aggreg_train(mod1, mod2, mod3, train_loader, .01, .1, 100, .5, .5)

table, loss = utils.aggreg_valid(mod1, mod2, mod3, X_valid, criterion, y_scaler)
print(f"Validation loss: {loss:.4f}")

utils.plot_residuals(table, 0, y_valid)