import tensorflow as tf
from time import time
import os
import matplotlib.pyplot as plt
from pathlib import Path

IMAGES_PATH = Path() / "images" / "generative"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

here = os.path.dirname(os.path.abspath(__file__))

from tensorflow.keras.models import load_model


def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = images.squeeze(axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

if __name__ == "__main__":
    variational_decoder = load_model(f'{here}/vae_fashion_mnist_decoder.h5', custom_objects={'Sampling': Sampling})
    codings_size = 10
    codings = tf.random.normal(shape=[3 * 7, codings_size])
    images = variational_decoder.predict(codings)
    plot_multiple_images(images, 7)
    save_fig(f"genai_{time()}", tight_layout=False)
    plt.show()

