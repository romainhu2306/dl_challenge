import torch
import torch.nn as nn

class baseline(nn.Module):
    '''
    Baseline model with 1 hidden layer with 512 neurons, dropout and batch normalization.
    SiLU activation function is used.

    - input_dim: number of features in the input ;
    - output_dim: number of features in the output.
    '''
    def __init__(self, input_dim, output_dim):
        super(baseline, self).__init__()
        self.L1 = nn.Linear(input_dim, 512)
        self.BN1 = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(.2)
        self.act = nn.SiLU()
        self.L2 = nn.Linear(512, output_dim)
    
    def forward(self, x):
        x = self.L1(x)
        x = self.BN1(x)
        x = self.drop(x)
        x = self.act(x)
        x = self.L2(x)
        return x


class overbase(nn.Module):
    '''
    Overparameterized model with 3 hidden layers with 1024, 512 and 512 neurons and
    batch normlization.
    SiLU activation function is used.

    - input_dim: number of features in the input ;
    - output_dim: number of features in the output.
    '''
    def __init__(self, input_dim, output_dim):
        super(overbase, self).__init__()
        self.L1 = nn.Linear(input_dim, 1024)
        self.BN1 = nn.BatchNorm1d(1024)
        self.act = nn.SiLU()
        self.L2 = nn.Linear(1024, 512)
        self.BN2 = nn.BatchNorm1d(512)
        self.L3 = nn.Linear(512, 512)
        self.BN3 = nn.BatchNorm1d(512)
        self.L4 = nn.Linear(512, output_dim)
    
    def forward(self, x):
        x = self.L1(x)
        x = self.BN1(x)
        x = self.act(x)
        x = self.L2(x)
        x = self.BN2(x)
        x = self.act(x)
        x = self.L3(x)
        x = self.BN3(x)
        x = self.act(x)
        x = self.L4(x)
        return x


class oversine(nn.Module):
    '''
    Overparameterized model with 3 hidden layers with 1024, 512 and 512 neurons and
    batch normlization.
    Sine activation function is used.

    - input_dim: number of features in the input ;
    - output_dim: number of features in the output.
    '''
    def __init__(self, input_dim, output_dim):
        super(oversine, self).__init__()
        self.L1 = nn.Linear(input_dim, 1024)
        self.BN1 = nn.BatchNorm1d(1024)
        self.L2 = nn.Linear(1024, 512)
        self.BN2 = nn.BatchNorm1d(512)
        self.L3 = nn.Linear(512, 512)
        self.BN3 = nn.BatchNorm1d(512)
        self.L4 = nn.Linear(512, output_dim)
    
    def forward(self, x):
        x = self.L1(x)
        x = self.BN1(x)
        x = torch.sin(x)
        x = self.L2(x)
        x = self.BN2(x)
        x = torch.sin(x)
        x = self.L3(x)
        x = self.BN3(x)
        x = torch.sin(x)
        x = self.L4(x)
        return x


class linear_aggreg(nn.Module):
    '''
    Returns a convex combination of the 5 outputs.
    Linear aggregator for 5 outputs.
    The weights are constrained to sum to 1 with a softmax function.

    - a1, a2, a3, a4, a5: initial weights for the 5 outputs.
    '''
    def __init__(self, a1, a2, a3, a4, a5):
        super(linear_aggreg, self).__init__()
        self.raw_coefs = nn.Parameter(torch.tensor([a1, a2, a3, a4, a5]), requires_grad = True)
    
    def forward(self, out1, out2, out3, out4, out5):
        coefs = torch.softmax(self.raw_coefs, dim = 0)
        x = coefs[0] * out1 + coefs[1] * out2 + coefs[2] * out3 + coefs[3] * out4 + coefs[4] * out5
        return x, coefs