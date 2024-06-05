import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tensorflow.keras import layers

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

# The first layer is the PixelCNN layer. This layer simply
# builds on the 2D convolutional layer, but includes masking.
# Next, we build our residual block layer.
# This is just a normal residual block, but based on the PixelConvLayer.

def get_pixelcnn(pixelcnn_input_shape, num_residual_blocks, num_pixelcnn_layers, num_embeddings):
    pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, num_embeddings)
    x = PixelConvLayer(
        mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
    )(ohe)

    for _ in range(num_residual_blocks):
        x = ResidualBlock(filters=128)(x)

    for _ in range(num_pixelcnn_layers):
        x = PixelConvLayer(
            mask_type="B",
            filters=128,
            kernel_size=1,
            strides=1,
            activation="relu",
            padding="valid",
        )(x)

    out = keras.layers.Conv2D(
        filters=num_embeddings, kernel_size=1, strides=1, padding="valid"
    )(x)

    pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
    pixel_cnn.summary()
    return pixel_cnn
