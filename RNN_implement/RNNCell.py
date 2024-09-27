import torch
import torch.nn as nn
import numpy as np

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, activation='tanh'):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        else:
            raise ValueError("activation function으로는 'relu' 또는 'tanh'를 사용하세요. Current : {activation}")
        
        self.W_xh = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.init_parameters()
    
    def init_parameters(self):
        std = 1.0/np.sqrt(self.hidden_size)
        nn.init.normal_(self.W_hh.weight, -std, std)
        nn.init.normal_(self.W_xh.weight, -std, std)

        if self.bias:
            nn.init.normal_(self.W_hh.bias, -std, std)
            nn.init.normal_(self.W_xh.bias, -std, std)
    
    def forward(self, x, h_prev):
        a_t = self.W_xh(x) + self.W_hh(h_prev)
        h_t = torch.tanh(a_t)
        return h_t
