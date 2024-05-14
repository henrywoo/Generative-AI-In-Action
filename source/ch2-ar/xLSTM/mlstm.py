import torch
import torch.nn as nn

class mLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size):
        super(mLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size  # Defines the size of the matrix memory

        # Gates initialization
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.memory_update = nn.Linear(input_dim + hidden_dim, memory_size**2)  # Update to memory matrix

    def forward(self, input, hidden):
        h_prev, C_prev = hidden
        combined = torch.cat((input, h_prev), 1)

        # Gates computations (unchanged)
        i_t = torch.sigmoid(self.input_gate(combined)).unsqueeze(-1).unsqueeze(-1)
        f_t = torch.sigmoid(self.forget_gate(combined)).unsqueeze(-1).unsqueeze(-1)
        o_t = torch.sigmoid(self.output_gate(combined))

        # Update memory matrix (unchanged)
        C_tilde = self.memory_update(combined).view(-1, self.memory_size, self.memory_size)

        # Cell state computation (unchanged)
        C_t = f_t * C_prev + i_t * C_tilde

        # 🔴 Key Change: Hidden state computation with proper broadcasting
        # Expand o_t to match the shape of C_t before multiplication
        h_t = (o_t.unsqueeze(-1).unsqueeze(-1) * torch.tanh(C_t)).mean(dim=[1, 2])

        return h_t, C_t


    def init_hidden(self, batch_size):
        # Initialize hidden state and matrix memory state with zeros
        return (torch.zeros(batch_size, self.hidden_dim),
                torch.zeros(batch_size, self.memory_size, self.memory_size))

# Example usage
batch_size = 1
input_dim = 10
hidden_dim = 20
memory_size = 5  # Size of the memory matrix

# Input tensor
input_tensor = torch.randn(batch_size, input_dim)

# Initialize the mLSTM
mlstm = mLSTM(input_dim, hidden_dim, memory_size)

# Initialize hidden state
hidden = mlstm.init_hidden(batch_size)

# Forward pass
output, new_state = mlstm(input_tensor, hidden)
print("Output:", output)
print("New Matrix Memory State:", new_state)
