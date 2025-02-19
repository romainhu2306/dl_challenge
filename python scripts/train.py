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
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
meteo = pd.read_parquet('meteo.parquet')
co2 = pd.read_csv('co2_emission_france.csv')


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
co2.rename(columns = {'Year':'year', 'CO2 emissions from fuel combustion, France':'co2'}, inplace = True)
co2.drop(columns = 'Units', inplace = True)
'''
CO2 emissions values are normalized and merged on year.
'''
scaler = StandardScaler()
co2['co2'] = scaler.fit_transform(co2[['co2']])

train['year'] = train['date'].dt.year
test['year'] = test['date'].dt.year

train = train.merge(co2, how = 'left', on = 'year')
test = test.merge(co2, how = 'left', on = 'year')

train.drop(columns = 'year', inplace = True)
test.drop(columns = 'year', inplace = True)


# Filling a few remaining NaNs in the test set.
test.interpolate(method = 'linear', limit_direction = 'both', inplace = True)


# Extracting temporal features.
train = utils.extract_date(train)
test = utils.extract_date(test)


# Adding days-off.
fr_holidays = holidays.France()
train['day_off'] = train['date'].apply(lambda x: x in fr_holidays).astype(int)
test['day_off'] = test['date'].apply(lambda x: x in fr_holidays).astype(int)


# Adding covid confinement.
train['covid'] = (train['date'] >= '2020-03-17') & (train['date'] <= '2021-05-03')
test['covid'] = (test['date'] >= '2020-03-17') & (test['date'] <= '2021-05-03')
train['covid'] = train['covid'].astype(int)
test['covid'] = test['covid'].astype(int)


# Dropping original date column.
train.drop(columns = 'date', inplace = True)
test.drop(columns = 'date', inplace = True)


