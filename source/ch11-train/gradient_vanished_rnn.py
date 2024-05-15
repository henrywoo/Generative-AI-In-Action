import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed for reproducibility
def set_seed(seed_value=42, has_np=False):
    import random
    if has_np:
        import numpy as np
        np.random.seed(seed_value)  # Numpy module
    random.seed(seed_value)  # Python random module
    torch.manual_seed(seed_value)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True  # CUDNN to be deterministic
        torch.backends.cudnn.benchmark = False



class Config:
    def __init__(self, seq_length, init_weight_range, num_layers, hidden_size, label, input_size=5):
        self.seq_length = seq_length
        self.init_weight_range = init_weight_range
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size  # Static for simplicity, can be made dynamic
        self.label = label


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, init_weight_range):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='tanh')
        self.initialize_weights(init_weight_range)

    def initialize_weights(self, init_weight_range):
        for p in self.rnn.parameters():
            nn.init.uniform_(p, a=-init_weight_range, b=init_weight_range)

    def forward(self, x):
        output, h_n = self.rnn(x)
        return output, h_n


def generate_data(seq_length, input_size, hidden_size):
    input = torch.randn(seq_length, 1, input_size)
    target = torch.randn(seq_length, 1, hidden_size)
    return input, target


def train_and_get_gradients(config):
    model = SimpleRNN(config.input_size, config.hidden_size, config.num_layers, config.init_weight_range)
    input, target = generate_data(config.seq_length, config.input_size, config.hidden_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    output, _ = model(input)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()

    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.norm().item())
    return gradients


def plot_all_gradients(configs, all_gradients):
    plt.figure(figsize=(10, 5))
    plt.style.use('ggplot')

    for config, gradients in zip(configs, all_gradients):
        plt.plot(gradients, marker='o', label=f"{config.label}")

    plt.title('Gradient Norms for Different Configurations')
    plt.xlabel('Layer Index')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig("gradient_vanished_rnn.png")
    plt.show()

if __name__ == '__main__':
    set_seed(2024, has_np=True)

    # Configurations for demonstration
    configs = [
        Config(seq_length=50, init_weight_range=0.02, num_layers=2, hidden_size=10, label='Config 1'),
        Config(seq_length=100, init_weight_range=0.02, num_layers=3, hidden_size=10, label='Config 2'),
        Config(seq_length=200, init_weight_range=0.01, num_layers=4, hidden_size=20, label='Config 3'),
        Config(seq_length=250, init_weight_range=0.01, num_layers=5, hidden_size=20, label='Config 4'),
        Config(seq_length=600, init_weight_range=0.01, num_layers=4, hidden_size=30, input_size=20, label='Config 5'),
    ]

    all_gradients = [train_and_get_gradients(config) for config in configs]
    plot_all_gradients(configs, all_gradients)