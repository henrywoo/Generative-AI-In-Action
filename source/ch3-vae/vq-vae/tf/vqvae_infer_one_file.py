import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tensorflow.keras import layers
class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta  # The `beta` parameter is best kept between [0.25, 2] as per the paper.

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Add losses
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "beta": self.beta
        })
        return config


class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, filters, kernel_size, strides=1, activation=None, padding='valid', **kwargs):
        super().__init__(**kwargs)  # Pass only generic layer arguments to superclass
        self.mask_type = mask_type
        # Initialize the Conv2D layer here with specific arguments
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            padding=padding
        )

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Create the mask based on the kernel shape
        kernel_shape = self.conv.kernel.shape
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, : kernel_shape[1] // 2, :, :] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, :, :] = 1.0

    def call(self, inputs):
        # Apply the mask to the kernel
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'mask_type': self.mask_type,
            'filters': self.conv.filters,
            'kernel_size': self.conv.kernel_size,
            'strides': self.conv.strides,
            'activation': self.conv.activation,
            'padding': self.conv.padding
        })
        return config

class ResidualBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = layers.Conv2D(filters=filters, kernel_size=1, activation='relu')
        self.pixel_conv = PixelConvLayer(
            mask_type='B', filters=filters // 2, kernel_size=3, activation='relu', padding='same'
        )
        self.conv2 = layers.Conv2D(filters=filters, kernel_size=1, activation='relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return layers.add([inputs, x])

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters
        })
        return config

# Load the VQ-VAE Model
vqvae_model = keras.models.load_model('vqvae_model.h5', custom_objects={'VectorQuantizer': VectorQuantizer})

# Load the PixelCNN Model
pixel_cnn = keras.models.load_model('pixel_cnn_model.h5', custom_objects={
    'PixelConvLayer': PixelConvLayer,
    'ResidualBlock': ResidualBlock
})
# Function to generate new images
def generate_new_images(pixel_cnn, vqvae_model, num_samples=10):
    # Assume `pixel_cnn` generates indices for VQ-VAE codes
    # Initialize priors for PixelCNN sampling
    priors = np.zeros((num_samples,) + pixel_cnn.input_shape[1:])
    sampler = keras.Model(inputs=pixel_cnn.input,
                          outputs=tfp.layers.DistributionLambda(tfp.distributions.Categorical)(pixel_cnn.output))

    # Generate one sample at a time due to the autoregressive nature
    for i in range(priors.shape[1]):
        for j in range(priors.shape[2]):
            probs = sampler.predict(priors)
            priors[:, i, j] = probs[:, i, j]

    # Quantize using trained embeddings from VQ-VAE
    decoder = vqvae_model.get_layer('decoder')
    quantizer = vqvae_model.get_layer('vector_quantizer')
    embeddings = quantizer.embeddings.numpy()
    priors_ohe = tf.one_hot(priors.astype('int32'), embeddings.shape[1])
    quantized = tf.matmul(priors_ohe, embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, (-1, 7, 7, 16))  # adjust shape based on your VQ-VAE architecture

    # Generate images
    generated_images = decoder.predict(quantized)

    # Plot the generated images
    for img in generated_images:
        plt.imshow(img.squeeze() + 0.5, cmap='gray')
        plt.axis('off')
        plt.show()


# Generate and display new images
generate_new_images(pixel_cnn, vqvae_model, num_samples=5)
