import torch
import torch.nn as nn
import numpy as np

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.output_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        if self.nonlinearity != 'tanh' or self.nonlinearity != 'relu':
            raise ValueError("nonlinearity function으로는 'relu' 또는 'tanh'를 사용하세요")
        
        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0/np.sqrt(self.hidden_size)
        for w in self.parameters():
            nn.init.uniform_(w,-std, std)
    
    def forward(self, input, hx=None):
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype = input.dtype, device=input.device)

        hy = self.x2h(input) + self.h2h(hx)

        if self.nonlinearity == 'tanh':
            hy = torch.tanh(hy)
        else:
            hy = torch.relu(hy)
        
        return hy