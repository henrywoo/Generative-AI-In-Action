import os
import sys

from scipy.linalg import sqrtm
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from torch.utils.data.distributed import DistributedSampler
from stanford_dogs import StanfordDogsDataset

here = os.path.abspath(os.path.dirname(__file__))


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
    if isinstance(images, list):
        images = torch.cat(images, dim=0)
    images = images.to(inception_model.device)
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    with torch.no_grad():
        preds = inception_model(images).softmax(dim=1).cpu().numpy()

    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits): ((i + 1) * preds.shape[0] // splits), :]
        kl_div = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl_div = np.mean(np.sum(kl_div, 1))
        scores.append(np.exp(kl_div))

    return np.mean(scores), np.std(scores)


class ImageNet1KDataSet:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']

        # Ensure the image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, sample['label']


def get_dataset(name, image_size, batch_size, data_dir, rank=None, world_size=None):
    if name == 'imagenet-1k':
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    try:
        if name == 'imagenet-1k':
            ds = load_dataset("imagenet-1k", trust_remote_code=True)
            train_dataset = ImageNet1KDataSet(ds["train"], transform=transform)
            test_dataset = ImageNet1KDataSet(ds["validation"], transform=transform)
        elif name == 'svhn':
            train_dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
            test_dataset = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)
        elif name == 'stanford_dogs':
            data_dir = os.getenv('DEVROOT2') + "/data/stanford_dogs"
            if not os.path.exists(data_dir):
                print("NO stanford_dogs data found")
                sys.exit(0)
            images_dir = os.path.join(data_dir, 'images/Images')
            annotations_dir = os.path.join(data_dir, 'annotations/Annotation')
            train_dataset = StanfordDogsDataset(images_dir=images_dir,
                                                annotations_dir=annotations_dir,
                                                transform=transform)
            test_dataset = StanfordDogsDataset(images_dir=images_dir,
                                               annotations_dir=annotations_dir,
                                               transform=transform)
        else:
            DatasetClass = getattr(datasets, name.upper())
            train_dataset = DatasetClass(root=data_dir, train=True, download=True, transform=transform)
            test_dataset = DatasetClass(root=data_dir, train=False, download=True, transform=transform)
    except AttributeError:
        raise ValueError(f"Dataset {name} not supported.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the dataset: {e}")

    if rank is not None and world_size is not None:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=8)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def available_datasets():
    from datasets import list_datasets

    available_datasets = list_datasets()
    return available_datasets


def check_pt():
    import os

    if not os.path.exists(f"{here}/pretrained_maskgit/VQGAN/last.ckpt"):
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id="llvictorll/Maskgit-pytorch", filename="pretrained_maskgit/VQGAN/last.ckpt",
                        local_dir=here)
        hf_hub_download(repo_id="llvictorll/Maskgit-pytorch", filename="pretrained_maskgit/VQGAN/model.yaml",
                        local_dir=here)


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


def save_checkpoint(state, filename=f'{here}/checkpoint.pth.tar'):
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return model, optimizer, start_epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return model, optimizer, 0, None


if __name__ == '__main__':
    demo_get_inception_score()
