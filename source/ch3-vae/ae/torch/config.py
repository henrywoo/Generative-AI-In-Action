import sys
from packaging import version
import sklearn

# Check Python version
assert sys.version_info >= (3, 7)

# Check scikit-learn version
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

# Check PyTorch version
import torch
import torchvision

# For demonstration purposes, let's check for PyTorch version 1.10.0 (you can adjust based on your requirements)
assert version.parse(torch.__version__) >= version.parse("1.10.0")

# Set up matplotlib configuration
import matplotlib.pyplot as plt

plt.rc('font', size=10)
plt.rc('axes', labelsize=10, titlesize=10)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

from pathlib import Path

# Create directory for saving images
IMAGES_PATH = Path() / "images" / "generative"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Check for GPU availability
if not torch.cuda.is_available():
    print("No GPU was detected. Neural nets can be very slow without a GPU.")
    if "google.colab" in sys.modules:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
    if "kaggle_secrets" in sys.modules:
        print("Go to Settings > Accelerator and select GPU.")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)