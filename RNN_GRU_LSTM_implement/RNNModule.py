import torch
import torch.nn as nn
from RNNCell import RNNCell

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bias=True, activation='tanh'):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bias = bias

        self.rnn_cells = nn.ModuleList()

        if activation == 'tanh':
            self.rnn_cells.append(RNNCell(self.input_size, self.hidden_size, self.bias, 'tanh'))
            for _ in range(1, self.num_layers):
                self.rnn_cells.append(RNNCell(self.hidden_size, self.hidden_size, self.bias, 'tanh'))
        
        elif activation == 'relu':
            self.rnn_cells.append(RNNCell(self.input_size, self.hidden_size, self.bias, 'relu'))
            for _ in range(1, self.num_layers):
                self.rnn_cells.append(RNNCell(self.hidden_size, self.hidden_size, self.bias, 'relu'))
        
        else:
            raise ValueError("Activation choosen different. Choose from 'tanh' or 'relu'")
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x, h0=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        if h0 is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        else:
            h = h0
        
        outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer] = self.rnn_cells[layer](input_t, h[layer])
                input_t = h[layer]
            outputs.append(h[-1])
        outputs = torch.stack(outputs, dim=1)
        out = self.fc(outputs[:,-1,:])

        return outputs, h