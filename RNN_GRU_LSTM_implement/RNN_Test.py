import torch
import torch.nn as nn
from RNNModule import SimpleRNN

# Define parameters
input_size = 10
hidden_size = 20
output_size = 1
batch_size = 5
seq_len = 7
num_layers = 2

# Create a batch of random input data (batch_size x seq_len x input_size)
x = torch.randn(batch_size, seq_len, input_size)

# Create instances of the custom and built-in RNNs
custom_rnn = SimpleRNN(input_size, hidden_size, output_size, num_layers)
builtin_rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

# Create initial hidden states
h0_custom = torch.zeros(num_layers, batch_size, hidden_size)
h0_builtin = torch.zeros(num_layers, batch_size, hidden_size)

# Forward pass through custom RNN
out_custom, h_n_custom = custom_rnn(x, h0_custom)

# Forward pass through built-in RNN
out_builtin, h_n_builtin = builtin_rnn(x, h0_builtin)

# Compare output shapes
print(f"\nCustom RNN Output Shape: {out_custom.shape}")
print(f"Built-in RNN Output Shape: {out_builtin.shape}\n")

print(f"Custom RNN Hidden State Shape: {h_n_custom.shape}")
print(f"Built-in RNN Hidden State Shape: {h_n_builtin.shape}")

print("\nAre output shapes equal?", out_custom.shape == out_builtin.shape)
print("Are hidden state shapes equal?", h_n_custom.shape == h_n_builtin.shape)

