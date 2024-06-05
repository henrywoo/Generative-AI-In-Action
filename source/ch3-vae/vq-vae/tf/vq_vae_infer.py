import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tensorflow.keras import layers
from vqvae import VectorQuantizer
from pixelcnn import PixelConvLayer, ResidualBlock

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
