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

# Getting device.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Loading files.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
meteo = pd.read_parquet('meteo.parquet')
co2 = pd.read_parquet('co2_emission_france.csv')


# Converting to datetime.
train['date'] = pd.to_datetime(train['date'], utc = True)
test['date'] = pd.to_datetime(test['date'], utc = True)
meteo['date'] = pd.to_datetime(meteo['date'], utc = True)


# Filling nan values for metropolises.
train['day_of_year'] = train['date'].dt.day_of_year
train['hour'] = train['date'].dt.hour

for col in train.columns:
  if col not in ['date', 'day_of_year', 'hour']:
    train[col] = train[col].fillna(train.groupby(['day_of_year', 'hour'])[col].transform('mean'))

train.drop(columns = ['day_of_year', 'hour'], inplace = True)


