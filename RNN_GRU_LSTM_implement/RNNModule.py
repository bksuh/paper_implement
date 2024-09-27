import torch
import torch.nn as nn
from rnncell import RNNCell

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, output_size=1, activation='tanh'):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        self.activation = activation

        self.rnn_cell_list = nn.ModuleList()

        if activation == 'tanh':
            self.rnn_cell_list.append(RNNCell(self.input_size, self.hidden_size, self.bias, 'tanh'))
            for _ in range(1, self.num_layers):
                self.rnn_cell_list.append(RNNCell(self.input_size, self.hidden_size, self.bias, 'tanh'))
        
        elif activation == 'relu':
            self.rnn_cell_list.append(RNNCell(self.input_size, self.output_size, self.bias, 'relu'))
            for _ in range(1, self.num_layers):
                self.rnn_cell_list.append(RNNCell(self.input_size, self.output_size, self.bias, 'relu'))
        
        else:
            raise ValueError("Activation choosen different. Choose from 'tanh' or 'relu'")
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, input, hx=None):
        batch_size = input.size(0)
        seq_len = input.size(1)

        if hx is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=input.device)
        else:
            h0 = hx
        
        outs = []
        hidden = list(h0.unbind(0))

        for t in range(seq_len):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:,t,:], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer-1], hidden[layer])
                
                hidden[layer] = hidden_l
            outs.append(hidden_l)
        out = outs[-1].squeeze()
        out = self.fc(out)

        return out