# https://keras.io/examples/generative/vq_vae/#pixelcnn-hyperparameters
# pip install -q tensorflow-probability
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf
from vqvae import get_vqvae, VectorQuantizer
from pixelcnn import PixelConvLayer, ResidualBlock, get_pixelcnn
import os
num_embeddings=128

class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            reconstructions = self.vqvae(x)
            # Calculate the losses.
            reconstruction_loss = tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            total_loss = reconstruction_loss + sum(self.vqvae.losses)
        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))
        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))
        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }


def show_subplot(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()


def display_codebook(trained_vqvae_model, vqvae_trainer, test_images):
    encoder = trained_vqvae_model.get_layer("encoder")
    quantizer = trained_vqvae_model.get_layer("vector_quantizer")

    encoded_outputs = encoder.predict(test_images)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

    for i in range(len(test_images)):
        plt.subplot(1, 2, 1)
        plt.imshow(test_images[i].squeeze() + 0.5)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(codebook_indices[i])
        plt.title("Code")
        plt.axis("off")
        plt.show()


def generate_imgs(pixel_cnn, quantizer, vqvae_trainer, encoded_outputs):
    # Create a mini sampler model.
    inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
    outputs = pixel_cnn(inputs, training=False)
    categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
    outputs = categorical_layer(outputs)
    sampler = keras.Model(inputs, outputs)

    # Create an empty array of priors.
    batch = 10
    priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
    batch, rows, cols = priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in range(rows):
        for col in range(cols):
            # Feed the whole array and retrieving the pixel value probabilities for the next pixel.
            probs = sampler.predict(priors)
            # Use the probabilities to pick pixel values and append the values to the priors.
            priors[:, row, col] = probs[:, row, col]

    print(f"Prior shape: {priors.shape}")
    # Perform an embedding lookup.
    pretrained_embeddings = quantizer.embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"), vqvae_trainer.num_embeddings).numpy()
    quantized = tf.matmul(priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))

    # Generate novel images.
    decoder = vqvae_trainer.vqvae.get_layer("decoder")
    generated_samples = decoder.predict(quantized)

    for i in range(batch):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i])
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated_samples[i].squeeze() + 0.5)
        plt.title("Generated Sample")
        plt.axis("off")
        plt.show()


def train_vqvae(epochs=10, f='model_vqvae.h5'):
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    x_train_scaled = (x_train / 255.0) - 0.5
    x_test_scaled = (x_test / 255.0) - 0.5
    data_variance = np.var(x_train / 255.0)
    #get_vqvae().summary()

    vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=num_embeddings)
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam())

    if not os.path.exists(f):
        vqvae_trainer.fit(x_train_scaled, epochs=epochs, batch_size=128)
        trained_vqvae_model = vqvae_trainer.vqvae
        idx = np.random.choice(len(x_test_scaled), 10)
        test_images = x_test_scaled[idx]
        reconstructions_test = trained_vqvae_model.predict(test_images)
        # Save VQ-VAE Model
        vqvae_trainer.vqvae.save(f)
        if 1:
            for test_image, reconstructed_image in zip(test_images, reconstructions_test):
                show_subplot(test_image, reconstructed_image)
            display_codebook(trained_vqvae_model, vqvae_trainer, test_images)
    else:
        # Load the saved VQ-VAE model
        vqvae_trainer.vqvae = keras.models.load_model(f, custom_objects={"VectorQuantizer": VectorQuantizer})
        trained_vqvae_model = vqvae_trainer.vqvae
        idx = np.random.choice(len(x_test_scaled), 10)
        test_images = x_test_scaled[idx]
    encoder = trained_vqvae_model.get_layer("encoder")
    quantizer = trained_vqvae_model.get_layer("vector_quantizer")

    # Generate the codebook indices.
    encoded_outputs = encoder.predict(x_train_scaled)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
    pixelcnn_input_shape = encoded_outputs.shape[1:-1]
    print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")
    return vqvae_trainer, quantizer, codebook_indices, pixelcnn_input_shape, encoded_outputs


if __name__ == "__main__":
    vqvae_trainer, quantizer, codebook_indices, pixelcnn_input_shape, encoded_outputs = train_vqvae(epochs=100)
    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")
    num_residual_blocks = 2
    num_pixelcnn_layers = 2
    pixel_cnn = get_pixelcnn(pixelcnn_input_shape, num_residual_blocks, num_pixelcnn_layers, num_embeddings)
    pixel_cnn.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    pixelcnn_model_path = 'model_pixelcnn.h5'
    if not os.path.exists(pixelcnn_model_path):
        pixel_cnn.fit(
            x=codebook_indices,
            y=codebook_indices,
            batch_size=128,
            epochs=100,
            validation_split=0.1,
        )
        pixel_cnn.save(pixelcnn_model_path)
    else:
        # Load the saved PixelCNN model
        pixel_cnn = keras.models.load_model(pixelcnn_model_path, custom_objects={"PixelConvLayer": PixelConvLayer,
                                                                                 "ResidualBlock": ResidualBlock})

    generate_imgs(pixel_cnn, quantizer, vqvae_trainer, encoded_outputs)


