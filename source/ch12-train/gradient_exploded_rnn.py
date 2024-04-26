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
    def __init__(self, learning_rate, std_dev, seq_length, label):
        self.input_size = 5
        self.hidden_size = 10
        self.num_layers = 2
        self.learning_rate = learning_rate
        self.std_dev = std_dev
        self.seq_length = seq_length
        self.label = label

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, std_dev):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='relu')
        self.initialize_weights(std_dev)

    def forward(self, x):
        output, h_n = self.rnn(x)
        return output, h_n

    def initialize_weights(self, std_dev):
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0, std=std_dev)

def generate_data(seq_length, input_size, hidden_size):
    # Create dummy input and target tensors
    input = torch.randn(seq_length, 1, input_size)
    target = torch.randn(seq_length, 1, hidden_size)
    return input, target

def train_model(model, input, target, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    output, _ = model(input)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    return model

def get_gradient_norms(model):
    gradient_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_norms.append(grad_norm)
    return gradient_norms

def plot_gradient_norms(configs, gradient_norms_dict):
    plt.figure(figsize=(10, 5))
    plt.style.use('ggplot')  # Use ggplot2 style
    for config in configs:
        plt.plot(gradient_norms_dict[config.label], marker='o', label=f"{config.label} (SeqLen={config.seq_length})")
    plt.title('Gradient Norms for Various Configurations and Sequence Lengths')
    plt.xlabel('Layer Index')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig('gradient_exploded_rnn.png')
    plt.show()


if __name__ == '__main__':
    set_seed(2024)
    # Define a set of configurations with different sequence lengths
    configs = [
        Config(learning_rate=0.01, std_dev=0.5, seq_length=20, label='Config 1'),
        Config(learning_rate=0.01, std_dev=0.4, seq_length=50, label='Config 2'),
        Config(learning_rate=0.02, std_dev=0.4, seq_length=200, label='Config 3'),
        # Add more configurations if needed
    ]

    gradient_norms_dict = {}

    for config in configs:
        model = SimpleRNN(config.input_size, config.hidden_size, config.num_layers, config.std_dev)
        input, target = generate_data(config.seq_length, config.input_size, config.hidden_size)
        model = train_model(model, input, target, config.learning_rate)
        gradient_norms = get_gradient_norms(model)
        gradient_norms_dict[config.label] = gradient_norms

    plot_gradient_norms(configs, gradient_norms_dict)