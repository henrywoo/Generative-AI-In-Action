from config import *
import tensorflow_probability as tfp
from vqvae_plot import *
from pixelcnn import get_pixelcnn



LATENT_DIM = 16
NUM_EMBEDDINGS = 128
BATCH_SIZE = 128

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
        print("commitment_loss == codebook_loss", commitment_loss == codebook_loss)
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
        self.loss_history = {
            "total_loss": [],
            "reconstruction_loss": [],
            "vqvae_loss": []
        }

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

        #self.loss_history["total_loss"].append(self.total_loss_tracker.result().numpy())
        #self.loss_history["reconstruction_loss"].append(self.reconstruction_loss_tracker.result().numpy())
        #self.loss_history["vqvae_loss"].append(self.vq_loss_tracker.result().numpy())

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

    '''def fit(self, x, epochs=1, batch_size=32, **kwargs):
        # Custom fit logic
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            # Reset the metrics at the start of the next epoch
            for metric in self.metrics:
                metric.reset_states()
            for step in range(0, len(x), batch_size):
                batch_x = x[step:step + batch_size]
                self.train_step(batch_x)
            # Optionally, print the current loss
            print(f"Total Loss: {self.total_loss_tracker.result().numpy()}, "
                  f"Reconstruction Loss: {self.reconstruction_loss_tracker.result().numpy()}, "
                  f"VQ-VAE Loss: {self.vq_loss_tracker.result().numpy()}")'''

def create_vqvae_model(latent_dim=16, num_embeddings=64):
    return VQVAE(latent_dim=latent_dim, num_embeddings=num_embeddings)

from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({'create_vqvae_model': create_vqvae_model})


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
        with keras.utils.custom_object_scope(
                {'VectorQuantizer': VectorQuantizer, 'VQVAE': VQVAE, 'create_vqvae_model': create_vqvae_model}):
            vqvae_trainer.vqvae = keras.models.load_model(model_file, custom_objects={
                'VQVAE': lambda: create_vqvae_model(LATENT_DIM, NUM_EMBEDDINGS)})
    else:
        vqvae_trainer.fit(x_train_scaled, epochs=30, batch_size=BATCH_SIZE)
        plot_training_losses(vqvae_trainer.loss_history)
        vqvae_trainer.vqvae.save(model_file)
    return vqvae_trainer, x_train_scaled, x_test_scaled

vqvae_trainer, x_train_scaled, x_test_scaled = load_or_train_vqvae()
trained_vqvae_model = vqvae_trainer.vqvae
trained_vqvae_model.summary()

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
plot_original_vs_code(test_images, codebook_indices)

########################################################################################
pixelcnn_input_shape = encoded_outputs.shape[1:-1]
print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")

pixel_cnn = get_pixelcnn(pixelcnn_input_shape, vqvae_trainer.vqvae.num_embeddings)
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
inputs = layers.Input(shape=pixelcnn_input_shape)
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
plot_code_vs_generated(priors, generated_samples)
