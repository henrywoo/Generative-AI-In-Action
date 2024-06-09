from config import *
import tensorflow_probability as tfp

# Define constants for model configuration
LATENT_DIM = 16
NUM_EMBEDDINGS = 64

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
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
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings**2, axis=0)
            - 2 * similarity
        )
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

class VQVAE(keras.Model):
    def __init__(self, latent_dim=16, num_embeddings=64, **kwargs):
        super(VQVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.encoder = self.build_encoder()
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_inputs = layers.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        encoder_outputs = layers.Conv2D(self.latent_dim, 1, padding="same")(x)
        return keras.Model(encoder_inputs, encoder_outputs, name="encoder")

    def build_decoder(self):
        latent_inputs = layers.Input(shape=(7, 7, self.latent_dim))  # Adjust input shape accordingly
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(latent_inputs)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same")(x)
        return keras.Model(latent_inputs, decoder_outputs, name="decoder")

    def call(self, inputs):
        encoder_outputs = self.encoder(inputs)
        quantized_latents = self.quantizer(encoder_outputs)
        reconstructions = self.decoder(quantized_latents)
        return reconstructions

class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.vqvae = VQVAE(self.latent_dim, self.num_embeddings)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
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
            reconstruction_loss = tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

def load_or_train_vqvae():
    model_path = Path("mbin/vqvae")
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / "vqvae_model"
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    x_train_scaled = (x_train / 255.0) - 0.5
    x_test_scaled = (x_test / 255.0) - 0.5
    data_variance = np.var(x_train / 255.0)
    vqvae_trainer = VQVAETrainer(data_variance, latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS)
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
    if model_file.exists():
        with keras.utils.custom_object_scope({'VectorQuantizer': VectorQuantizer, 'VQVAE': VQVAE}):
            vqvae_trainer.vqvae = keras.models.load_model(model_file)
    else:
        vqvae_trainer.fit(x_train_scaled, epochs=3, batch_size=128)
        vqvae_trainer.vqvae.save(model_file)
    return vqvae_trainer, x_train_scaled, x_test_scaled

def show_all_subplots(originals, reconstructions):
    num_images = len(originals)
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.2, 3))

    for i in range(num_images):
        axes[0, i].imshow(originals[i].squeeze() + 0.5)
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructions[i].squeeze() + 0.5)
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

vqvae_trainer, x_train_scaled, x_test_scaled = load_or_train_vqvae()
trained_vqvae_model = vqvae_trainer.vqvae

idx = np.random.choice(len(x_test_scaled), 10)
test_images = x_test_scaled[idx]
reconstructions_test = trained_vqvae_model.predict(test_images)
show_all_subplots(test_images, reconstructions_test)

encoder = vqvae_trainer.vqvae.encoder
quantizer = vqvae_trainer.vqvae.quantizer
encoded_outputs = encoder.predict(test_images)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])


# Plot the original and code images
fig, axs = plt.subplots(2, len(test_images), figsize=(15, 3))
for i in range(len(test_images)):
    axs[0, i].imshow(test_images[i].squeeze() + 0.5)
    axs[0, i].set_title("Original")
    axs[0, i].axis("off")

    axs[1, i].imshow(codebook_indices[i])
    axs[1, i].set_title("Code")
    axs[1, i].axis("off")

plt.tight_layout()
plt.show()

## PixelCNN hyperparameters
num_residual_blocks = 2
num_pixelcnn_layers = 2
pixelcnn_input_shape = encoded_outputs.shape[1:-1]
print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")

class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super().__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)
class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])


pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
ohe = tf.one_hot(pixelcnn_inputs, vqvae_trainer.num_embeddings)
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
    filters=vqvae_trainer.num_embeddings, kernel_size=1, strides=1, padding="valid"
)(x)

pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
pixel_cnn.summary()

# Generate the codebook indices.
encoded_outputs = encoder.predict(x_train_scaled)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)

codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")
model_path = Path("mbin/vqvae")
model_path.mkdir(parents=True, exist_ok=True)
model_file = model_path / "pixel_cnn_model"
if model_file.exists():
    pixel_cnn = keras.models.load_model(model_file, custom_objects={"tfp": tfp})
else:
    pixel_cnn.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    pixel_cnn.fit(
        x=codebook_indices,
        y=codebook_indices,
        batch_size=128,
        epochs=30,
        validation_split=0.1,
    )
    pixel_cnn.save(model_file)
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
        # Feed the whole array and retrieving the pixel value probabilities for the next
        # pixel.
        probs = sampler.predict(priors)
        # Use the probabilities to pick pixel values and append the values to the priors.
        priors[:, row, col] = probs[:, row, col]

print(f"Prior shape: {priors.shape}")
# Perform an embedding lookup.
pretrained_embeddings = quantizer.embeddings
priors_ohe = tf.one_hot(priors.astype("int32"), vqvae_trainer.num_embeddings).numpy()
quantized = tf.matmul(
    priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
)
quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))

# Generate novel images.
decoder = vqvae_trainer.vqvae.get_layer("decoder")
generated_samples = decoder.predict(quantized)

# Assuming `priors` and `generated_samples` are defined and have the same batch size

num_images = len(priors)
fig, axs = plt.subplots(2, num_images, figsize=(num_images * 1.2, 2.5))

for i in range(num_images):
    # Plot the code image
    axs[0, i].imshow(priors[i])
    axs[0, i].set_title("Code")
    axs[0, i].axis("off")

    # Plot the generated sample
    axs[1, i].imshow(generated_samples[i].squeeze() + 0.5)
    axs[1, i].set_title("Generated Sample")
    axs[1, i].axis("off")

plt.tight_layout()
save_fig("vqvae_generated_samples")
plt.show()
