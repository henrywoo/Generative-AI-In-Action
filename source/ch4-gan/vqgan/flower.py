"""
This is to get the mean and std of the Oxford 102 Flower Dataset
"""
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Define the transform without normalization
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor(),  # Convert the image to a tensor
])

# Load the dataset
train_data = datasets.Flowers102(root='data', split='train', download=True, transform=transform)

# Create a DataLoader
data_loader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=4)

# Function to compute mean and std
def compute_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std

# Compute the mean and std
mean, std = compute_mean_std(data_loader)

print(f'Mean: {mean}')
print(f'Std: {std}')
