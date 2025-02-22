import pandas as pd
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
    df = meteo
    df[var_name] = scaler.fit_transform(df[[var_name]])

    # Pivoting feature by station.
    var = df[['date', var_name, 'numer_sta']]
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


def simple_train(model, train_loader, criterion, learning_rate, num_epochs, device, scheduler = False):
    '''
    Train a simple non-aggregated model on the train set.
    '''
    model.train()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    if scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 0, factor = .1)

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


def simple_valid(model, X_valid, y_valid, criterion, scaler):
    '''
    Evaluates a simple non-aggregated model on the validation set.
    '''
    nrows = len(X_valid)
    table = torch.empty(nrows, 25)
    model.eval()
    with torch.no_grad():
        for i in range(nrows):
            x = X_valid[i].unsqueeze(0)
            x = model(x)
            table[i] = x[0]
    table = torch.tensor(scaler.inverse_transform(table), dtype = torch.float32)
    loss = criterion(table, y_valid)
    loss = np.sqrt(loss.item())
    return table, loss


def aggreg_train(model1, model2, model3, train_loader, criterion, learning_rate, num_epochs):
    '''
    Train an aggregated model on the train set.
    '''
    model1.train()
    model2.train()
    model3.train()
    opti1 = optim.Adam(model1.parameters(), lr = learning_rate)
    opti2 = optim.Adam(model2.parameters(), lr = learning_rate)
    opti3 = optim.Adam(model3.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        ep_loss = .0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            opti1.zero_grad()
            opti2.zero_grad()
            opti3.zero_grad()

            out1 = model1(X)
            out2 = model2(X)
            out12 = torch.cat((out1, out2), dim = 1)
            out3 = model3(out12)
        
            loss = criterion(out3, y)
            loss.backward()

            opti1.step()
            opti2.step()
            opti3.step()

            ep_loss += loss.item()
        ep_loss = np.sqrt(ep_loss/len(train_loader))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {ep_loss:.4f}")
    
    return model1, model2, model3


def competitive_aggreg_train(model1, model2, model3, model4, model5, aggreg, train_loader, lr1, lr2, num_epochs):
    '''
    Trains the competitive aggregation model with 5 competitors.
    Each aggregated model tries to maximize it is given by the linear aggregator.
    This function also plots the trajectories of the weights for each model.
    '''
    model1.train()
    model2.train()
    model3.train()
    model4.train()
    model5.train()
    aggreg.train()
    opti1 = optim.Adam(model1.parameters(), lr = lr1)
    opti2 = optim.Adam(model2.parameters(), lr = lr1)
    opti3 = optim.Adam(model3.parameters(), lr = lr1)
    opti4 = optim.Adam(model4.parameters(), lr = lr1)
    opti5 = optim.Adam(model5.parameters(), lr = lr1)
    opti_aggreg = optim.Adam(aggreg.parameters(), lr = lr2)

    mse = nn.MSELoss()
    mae = nn.L1Loss()
    target_weight = torch.tensor(1, dtype = torch.float32)
    coefs_list1 = []
    coefs_list2 = []
    coefs_list3 = []
    coefs_list4 = []
    coefs_list5 = []

    for epoch in range(num_epochs):
        ep_loss = .0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            opti1.zero_grad()
            opti2.zero_grad()
            opti3.zero_grad()
            opti4.zero_grad()
            opti5.zero_grad()
            opti_aggreg.zero_grad()

            out1 = model1(X)
            out2 = model2(X)
            out3 = model3(X)
            out4 = model4(X)
            out5 = model5(X)
            out_aggreg = aggreg(out1, out2, out3, out4, out5)
            output = out_aggreg[0]
            coefs = out_aggreg[1]
        
            loss5 = mae(coefs[4], target_weight)
            loss5.backward(retain_graph = True)
            opti5.step()

            loss4 = mae(coefs[3], target_weight)
            loss4.backward(retain_graph = True)
            opti4.step()

            loss3 = mae(coefs[2], target_weight)
            loss3.backward(retain_graph = True)
            opti3.step()

            loss2 = mae(coefs[1], target_weight)
            loss2.backward(retain_graph = True)
            opti2.step()

            loss1 = mae(coefs[0], target_weight)
            loss1.backward(retain_graph = True)
            opti1.step()

            loss_aggreg = mse(output, y)
            loss_aggreg.backward()
            opti_aggreg.step()

            opti1.step()
            opti2.step()
            opti3.step()

            ep_loss += loss.item()
        ep_loss = np.sqrt(ep_loss/len(train_loader))
    
    X_axis = np.arange(0, len(coefs_list1))
    plt.plot(X_axis, coefs_list1, label = 'a1')
    plt.plot(X_axis, coefs_list2, label = 'a2')
    plt.plot(X_axis, coefs_list3, label = 'a3')
    plt.plot(X_axis, coefs_list4, label = 'a4')
    plt.plot(X_axis, coefs_list5, label = 'a5')
    plt.legend()
    plt.show()
    
    return model1, model2, model3, model4, model5


def aggreg_valid(model1, model2, model3, X_valid, y_valid, criterion, scaler):
    '''
    Evaluates an aggregation model on the validation set.
    '''
    nrows = len(X_valid)
    table = torch.empty(nrows, 25)
    model.eval()
    with torch.no_grad():
        for i in range(nrows):
            x = X_valid[i].unsqueeze(0)
            x1 = model1(x)
            x2 = model2(x)
            x12 = torch.cat((x1, x2), dim = 1)
            x3 = model3(x12)
            table[i] = x3[0]
    table = torch.tensor(scaler.inverse_transform(table), dtype = torch.float32)
    loss = criterion(table, y_valid)
    loss = np.sqrt(loss.item())
    return table, loss


def plot_residuals(table, col, y):
    '''
    Plots the residuals from a validation.
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


def OL_plus_MSE(model1, model2, alpha1, alpha2, X, y):
    '''
    MSE loss + Orthogonal loss.
    '''
    OL = OL(model1, model2, alpha1, alpha2)
    MSE = nn.MSELoss(X, y)
    Loss = OL + MSE
    return Loss


def extract_date(df, date_col = 'date'):
    '''
    Extracts temporal features and applies sine embedding.
    '''
    df['day_of_year'] = np.sin(2*np.pi*df[date_col].dt.day_of_year/365)
    df['day_of_week'] = np.sin(2*np.pi*df[date_col].dt.day_of_week/7)
    df['hour'] = np.sin(2*np.pi*df[date_col].dt.hour/24)
    df['minute'] = np.sin(2*np.pi*df[date_col].dt.minute/60)
    return df