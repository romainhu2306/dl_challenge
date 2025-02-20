import pandas
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def add_meteo_var(var_name, suffix, train, test, meteo):
    '''
    Adds a given meteorological variable to the train set.
    The variables are pivoted by station, normalized and upsampled to half-hour frequency.
    '''
    scaler = StandardScaler()
    meteo[var_name] = scaler.fit_transform(meteo[[var_name]])

    # Pivoting feature by station.
    var = meteo[['date', var_name, 'numer_sta']]
    var = meteo.pivot_table(var, index = 'date', values = var_name, columns = 'numer_sta')

    # Upsampling to half-hour frequency.
    var = var.resample('30min').interpolate(method = 'linear', limit_direction = 'both')

    # Dividing train and test sets.
    var_train = var.loc['2017-02-13 00:30:00+00:00':'2021-12-31 22:30:00+00:00'].reset_index()
    var_test = var.loc['2021-12-31 23:00:00+00:00':].reset_index()

    # Merging with train and test sets.
    train = train.merge(var_train, on = 'date', how = 'left', suffixes = ('', suffix))
    test = test.merge(var_test, on = 'date', how = 'left', suffixes = ('', suffix))

    return train, test


def simple_train(model, train_loader, criterion, learning_rate, num_epochs):
    '''
    Train a simple non-aggregated model on the train set.
    '''
    model.train()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        ep_loss = .0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        ep_loss = np.sqrt(ep_loss/len(train_loader))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {ep_loss:.4f}")
    
    return model


def simple_valid(model, valid_set, criterion, scaler):
    '''
    Evaluates a simple non-aggregated model on the validation set.
    '''
    nrows = len(valid_set)
    table = torch.empty(nrows, 25)
    model.eval()
    with torch.no_grad():
        for i in range(nrows):
            x = valid_set[i].unsqueeze(0)
            x = model(x)
            table[i] = x[0]
    table = torch.tensor(scaler.inverse_transform(table), dtype = torch.float32)
    loss = criterion(table, valid_set)
    loss = np.sqrt(loss.item())
    return table, loss


def aggreg_valid(model1, model2, model3, valid_set, criterion, scaler):
    '''
    Evaluates an aggregation model on the validation set.
    '''
    nrows = len(valid_set)
    table = torch.empty(nrows, 25)
    model.eval()
    with torch.no_grad():
        for i in range(nrows):
            x = valid_set[i].unsqueeze(0)
            x1 = model1(x)
            x2 = model2(x)
            x12 = torch.cat((x1, x2), dim = 1)
            x3 = model3(x12)
            table[i] = x3[0]
    table = torch.tensor(scaler.inverse_transform(table), dtype = torch.float32)
    loss = criterion(table, valid_set)
    loss = np.sqrt(loss.item())
    return table, loss


def plot_residuals(table, col, y):
    '''
    Plots the residuals from an evaluation.
    '''
    nrows = len(table)
    X = np.arange(nrows)
    residuals = y[:, col] - table[:, col]
    plt.plot(X, residuals)
    plt.axhline(0, color = 'red')
    plt.title('Residuals')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.show()


def OL(model1, model2, alpha1, alpha2, eps = 1e-8):
    '''
    Orthogonal loss function.
    '''
    params1 = torch.cat([p.view(-1) for p in model1.parameters()])
    params2 = torch.cat([p.view(-1) for p in model2.parameters()])

    dot = torch.dot(params1, params2)
    norm1 = torch.norm(params1)
    norm2 = torch.norm(params2)

    loss = alpha1*torch.abs(dot) + alpha2/(norm1 + eps) + alpha2/(norm2 + eps)
    return loss


def extract_date(df, date_col = 'date'):
    '''
    Extracts temporal features and applies sine embedding.
    '''
    df['day_of_year'] = np.sin(2*np.pi*df[date_col].dt.day_of_year/365)
    df['day_of_week'] = np.sin(2*np.pi*df[date_col].dt.day_of_week/7)
    df['hour'] = np.sin(2*np.pi*df[date_col].dt.hour/24)
    df['minute'] = np.sin(2*np.pi*df[date_col].dt.minute/60)
    return df