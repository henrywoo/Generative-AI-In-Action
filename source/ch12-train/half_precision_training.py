import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os


# Define a larger model
class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def train_model(fp16=False):
    # Create model, loss function, and optimizer
    model = LargeModel().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Create a GradScaler object if using FP16
    scaler = GradScaler() if fp16 else None

    # Example training loop
    for epoch in range(1):  # loop over the dataset multiple times
        inputs = torch.randn(64, 1024).cuda()
        targets = torch.randn(64, 10).cuda()

        # Forward pass with autocast if using FP16
        if fp16:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()

    return model


def save_model_weights(model, filename):
    # Save the model parameters
    torch.save(model.state_dict(), filename)


def load_model(filepath, fp16=False):
    # Create a new model instance
    model = LargeModel().cuda()
    # Load the state dictionary
    state_dict = torch.load(filepath)
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    # If loading FP16 model, ensure the model is in FP16 mode
    if fp16:
        model.half()
    return model


if __name__ == '__main__':
    # Train and save FP16 model
    model_fp16 = train_model(fp16=True)
    model_fp16.half()
    save_model_weights(model_fp16, 'large_model_fp16_weights.pth')
    print("Converted FP16 model saved.")

    # Train and save FP32 model
    model_fp32 = train_model(fp16=False)
    save_model_weights(model_fp32, 'large_model_fp32_weights.pth')
    print("FP32 model saved.")

    # Get the size of the saved models
    fp16_model_size = os.path.getsize('large_model_fp16_weights.pth')
    fp32_model_size = os.path.getsize('large_model_fp32_weights.pth')

    print(f"Converted FP16 model size: {fp16_model_size / 1024:.2f} KB")
    print(f"FP32 model size: {fp32_model_size / 1024:.2f} KB")

    # Load the FP16 model
    loaded_model_fp16 = load_model('large_model_fp16_weights.pth', fp16=True)
    print("FP16 model loaded.")

    # Load the FP32 model
    loaded_model_fp32 = load_model('large_model_fp32_weights.pth', fp16=False)
    print("FP32 model loaded.")

    # Test the loaded models (e.g., perform a forward pass)
    sample_input = torch.randn(64, 1024).cuda()

    with torch.no_grad():
        output_fp16 = loaded_model_fp16(sample_input.half())  # Ensure input is in FP16
        output_fp32 = loaded_model_fp32(sample_input)  # Input is in FP32

    print("FP16 model output:", output_fp16.shape)
    print("FP32 model output:", output_fp32.shape)


"""
Output:
-------------------------------------------
Converted FP16 model saved.
FP32 model saved.
Converted FP16 model size: 1367.10 KB
FP32 model size: 2730.22 KB
FP16 model loaded.
FP32 model loaded.
FP16 model output: torch.Size([64, 10])
FP32 model output: torch.Size([64, 10])

Explanation:
-------------------------------------------
The significant size difference between the FP16 and FP32 model files indicates that the FP16 model indeed uses half the precision, resulting in a smaller file size. Additionally, the model outputs have the correct shape, confirming that the models are working as intended.
"""