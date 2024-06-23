from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


def create_mnist_dataset(data_path, batch_size, **kwargs):
    train = kwargs.get("train", True)  # Flag to load training or testing split
    download = kwargs.get("download", True)  # Download if dataset isn't found

    # Load the MNIST dataset and apply transformations
    dataset = MNIST(root=data_path, train=train, download=download, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, ), (0.5, ))  # Normalize for better training
    ]))

    # Parameters for the DataLoader
    loader_params = dict(
        shuffle=kwargs.get("shuffle", True),  # Shuffle data during training
        drop_last=kwargs.get("drop_last", True),  # Drop incomplete batches
        pin_memory=kwargs.get("pin_memory", True),  # Use pinned memory for GPU
        num_workers=kwargs.get("num_workers", 4),  # Number of data loading processes
    )

    # Create the DataLoader object
    dataloader = DataLoader(dataset, batch_size=batch_size, **loader_params)

    return dataloader