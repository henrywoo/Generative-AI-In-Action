import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
    def __init__(self, init_weight_range, num_layers, label):
        self.init_weight_range = init_weight_range
        self.num_layers = num_layers
        self.label = label


class DeepNet(nn.Module):
    def __init__(self, num_layers, init_weight_range):
        super(DeepNet, self).__init__()
        self.layers = nn.ModuleList()
        input_size = 10  # Input size for the first layer
        for i in range(num_layers):
            output_size = 10 if i == num_layers - 1 else 10  # Set the last layer's output size to 10
            self.layers.append(nn.Linear(input_size, output_size))
            self.layers.append(nn.Sigmoid())
            self.initialize_weights(init_weight_range, i)
            input_size = 10  # Input size for subsequent layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def initialize_weights(self, init_weight_range, index):
        layer = self.layers[index * 2]  # Accessing the Linear layer
        nn.init.uniform_(layer.weight, a=-init_weight_range, b=init_weight_range)


def generate_data():
    input = torch.randn(10, 10)  # Input size is fixed at 10
    target = torch.randn(10, 10)  # Output size is fixed at 10
    return input, target


def train_and_get_gradients(config):
    model = DeepNet(num_layers=config.num_layers, init_weight_range=config.init_weight_range)
    input, target = generate_data()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    output = model(input)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()

    gradients = []
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            gradients.append(layer.weight.grad.norm().item())
    return gradients


def plot_all_gradients(configs, all_gradients):
    plt.figure(figsize=(10, 5))
    plt.style.use('ggplot')

    for config, gradients in zip(configs, all_gradients):
        plt.plot(gradients, marker='o', label=f"{config.label}")

    plt.title('Gradient Norms Across Layers')
    plt.xlabel('Layer')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig("gradient_vanished_deepnet.png")
    plt.show()


if __name__ == '__main__':
    set_seed(2024, has_np=True)

    # Configurations for demonstration
    configs = [
        Config(init_weight_range=0.1, num_layers=20, label='Config 1'),
        Config(init_weight_range=0.1, num_layers=30, label='Config 2'),
        Config(init_weight_range=0.01, num_layers=20, label='Config 3'),
    ]

    all_gradients = [train_and_get_gradients(config) for config in configs]
    plot_all_gradients(configs, all_gradients)
