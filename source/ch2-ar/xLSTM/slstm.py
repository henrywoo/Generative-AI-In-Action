"""
Here's a simple implementation of an sLSTM (scalar LSTM) class using PyTorch. This class will demonstrate the core
elements of an LSTM, specifically focusing on the scalar aspect. We'll include an input gate, forget gate, output
gate, and a simple scalar memory cell without implementing the full complexity of exponential gating or advanced
memory structures like those in the xLSTM architecture.
"""
import torch
import torch.nn as nn
"""
This code defines a basic structure for an sLSTM layer, suitable for processing sequences when integrated into a loop 
over timesteps. The forward method takes the current input and the previous hidden state (which includes the hidden 
state and the cell state) and calculates the new hidden state and cell state. This can be iterated over a sequence of 
inputs to process data recurrently.
"""

class sLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(sLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Gates initialization
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.cell_state = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, input, hidden):
        h_prev, c_prev = hidden

        # Combined input
        combined = torch.cat((input, h_prev), 1)

        # LSTM gates computations
        i_t = torch.sigmoid(self.input_gate(combined))
        f_t = torch.sigmoid(self.forget_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))

        # Cell state computation
        c_t = f_t * c_prev + i_t * torch.tanh(self.cell_state(combined))

        # Hidden state computation
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

    def init_hidden(self, batch_size):
        # Initialize hidden and cell states with zeros
        return (torch.zeros(batch_size, self.hidden_dim),
                torch.zeros(batch_size, self.hidden_dim))


# Example usage
batch_size = 1
input_dim = 10
hidden_dim = 20

# Input tensor
input_tensor = torch.randn(batch_size, input_dim)

# Initialize the sLSTM
slstm = sLSTM(input_dim, hidden_dim)

# Initialize hidden state
hidden = slstm.init_hidden(batch_size)

# Forward pass
output, new_state = slstm(input_tensor, hidden)
print("Output:", output)
print("New Hidden State:", new_state)
