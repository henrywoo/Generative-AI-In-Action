from scipy.linalg import sqrtm
import numpy as np
import torch.nn.functional as F
import torch
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import inception_v3


def calculate_fid(real_features, fake_features):
    # Calculate mean and covariance statistics
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # Calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # Calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def get_inception_score(images, inception_model, splits=10):
    images = torch.cat(images, 0)
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    with torch.no_grad():
        preds = inception_model(images).softmax(dim=1).cpu().numpy()

    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits) : ((i + 1) * preds.shape[0] // splits), :]
        kl_div = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl_div = np.mean(np.sum(kl_div, 1))
        scores.append(np.exp(kl_div))

    return np.mean(scores), np.std(scores)


def get_dataset(name, image_size, batch_size, data_dir):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    try:
        if name in ['imagenet']:
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            train_dataset = ImageFolder(root=f'{data_dir}/train', transform=transform)
            test_dataset = ImageFolder(root=f'{data_dir}/val', transform=transform)
        else:
            dataset_class = eval(f"datasets.{name.upper()}")
            if name == 'svhn':
                train_dataset = dataset_class(root=data_dir, split='train', download=True, transform=transform)
                test_dataset = dataset_class(root=data_dir, split='test', download=True, transform=transform)
            else:
                train_dataset = dataset_class(root=data_dir, train=True, download=True, transform=transform)
                test_dataset = dataset_class(root=data_dir, train=False, download=True, transform=transform)

    except AttributeError:
        raise ValueError(f"Dataset {name} not supported.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def available_datasets():
    from datasets import list_datasets

    available_datasets = list_datasets()
    return available_datasets


def check_pt():
    import os

    if not os.path.exists("VQGAN/last.ckpt"):
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id="llvictorll/Maskgit-pytorch", filename="pretrained_maskgit/VQGAN/last.ckpt", local_dir=".")
        hf_hub_download(repo_id="llvictorll/Maskgit-pytorch", filename="pretrained_maskgit/VQGAN/model.yaml", local_dir=".")

def get_inception_score(images, inception_model, splits=10):
    # Ensure the input images are in a single batch
    if isinstance(images, list):
        images = torch.stack(images, dim=0)

    # Resize images to the required input size of Inception model
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

    # Ensure the images are in the correct shape (B, C, H, W)
    if images.dim() == 3:
        images = images.unsqueeze(0)

    with torch.no_grad():
        preds = inception_model(images).softmax(dim=1).cpu().numpy()

    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits): ((i + 1) * preds.shape[0] // splits), :]
        kl_div = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl_div = np.mean(np.sum(kl_div, 1))
        scores.append(np.exp(kl_div))

    return np.mean(scores), np.std(scores)


import torch
from torchvision.models import inception_v3
import torch.nn.functional as F


def demo_get_inception_score():
    # Create a batch of dummy images (batch_size, channels, height, width)
    batch_size = 16
    channels = 3
    height = 256
    width = 256
    dummy_images = torch.randn(batch_size, channels, height, width)

    # Load pretrained Inception v3 model
    inception_model = inception_v3(pretrained=True, transform_input=False).eval()

    # Move model and images to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_images = dummy_images.to(device)
    inception_model = inception_model.to(device)

    # Calculate Inception Score
    mean, std = get_inception_score(dummy_images, inception_model)
    print(f'Inception Score: {mean} ± {std}')


# Run the test function

if __name__ == '__main__':
    demo_get_inception_score()
