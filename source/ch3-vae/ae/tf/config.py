import sys
import os
assert sys.version_info >= (3, 7)

from packaging import version
import sklearn


def set_seed(seed=42):
    import sys

    try:
        import random
        random.seed(seed)
    except ImportError as e:
        print(f"Error importing random: {e}", file=sys.stderr)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError as e:
        print(f"Error importing numpy: {e}", file=sys.stderr)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError as e:
        print(f"Error importing tensorflow: {e}", file=sys.stderr)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError as e:
        print(f"Error importing torch: {e}", file=sys.stderr)

    print("Seed set to", seed)

set_seed()

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

assert version.parse(tf.__version__) >= version.parse("2.8.0")

import matplotlib.pyplot as plt

plt.rc('font', size=10)
plt.rc('axes', labelsize=10, titlesize=10)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

from pathlib import Path

IMAGES_PATH = Path() / "images" / "generative"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. Neural nets can be very slow without a GPU.")
    if "google.colab" in sys.modules:
        print("Go to Runtime > Change runtime and select a GPU hardware "
              "accelerator.")
    if "kaggle_secrets" in sys.modules:
        print("Go to Settings > Accelerator and select GPU.")

import numpy as np
import matplotlib as mpl


