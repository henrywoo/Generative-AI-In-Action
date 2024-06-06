import torch
import torch.nn.functional as F
import random

random.seed(0)
torch.manual_seed(0)

# Define the perceptron class
class MyDNN(torch.nn.Module):
    def __init__(self):
        super(MyDNN, self).__init__()
        # Define a single linear layer that takes 3 inputs and produces 1 output
        self.layer1 = torch.nn.Linear(3, 1)

    def forward(self, x):
        # Forward pass: input data is passed through the linear layer
        x = self.layer1(x)
        # Apply ReLU activation function
        x = F.relu(x)
        return x


# Function to show a simple demo
def demo_perceptron():
    # Create an instance of the perceptron
    perceptron = MyDNN()

    # Print the initial weights and biases of the perceptron
    print("Initial weights and biases:")
    print(perceptron.layer1.weight)
    print(perceptron.layer1.bias)

    # Create a simple input tensor (3 features)
    input_data = torch.tensor([1.0, 2.0, 3.0])

    # Pass the input data through the perceptron to get the output
    output = perceptron(input_data)

    # Print the input and output
    print("\nInput data:")
    print(input_data)
    print("Output data:")
    print(output)



if __name__ == '__main__':
    demo_perceptron()
