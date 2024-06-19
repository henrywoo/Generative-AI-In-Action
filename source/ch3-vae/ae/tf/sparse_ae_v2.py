import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from stacked_ae import X_train, X_valid, plot_reconstructions, save_fig

"""
Model: "sequential_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 sequential_3 (Sequential)   (None, 300)               108800    

 sequential_4 (Sequential)   (None, 28, 28)            109284    

=================================================================
Total params: 218,084
Trainable params: 218,084
Non-trainable params: 0
_________________________________________________________________
By default, the weights in Keras layers are initialized using the "Glorot Uniform" initializer, also 
known as Xavier uniform initializer. If you want to specify a different initializer, you can do so 
by adding the kernel_initializer parameter to your Dense layers.
"""


def create_sparse_l1_autoencoder():
    encoder = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(300, activation="sigmoid"),
        tf.keras.layers.ActivityRegularization(l1=1e-4)
    ])
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(28 * 28),
        tf.keras.layers.Reshape([28, 28])
    ])
    autoencoder = tf.keras.Sequential([encoder, decoder])
    return autoencoder


def train_autoencoder(autoencoder, X_train, X_valid, checkpoint_path, epochs=10):
    autoencoder.compile(loss="mse", optimizer="nadam")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    history = autoencoder.fit(X_train, X_train, epochs=epochs,
                              validation_data=(X_valid, X_valid),
                              batch_size=64,
                              callbacks=[checkpoint_callback])
    return history


def plot_sparsity_loss():
    p = 0.1
    q = np.linspace(0.001, 0.999, 500)
    kl_div = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    mse = (p - q) ** 2
    mae = np.abs(p - q)
    plt.plot([p, p], [0, 0.3], "k:")
    plt.text(0.05, 0.32, "Target\nsparsity", fontsize=14)
    plt.plot(q, kl_div, "b-", label="KL divergence")
    plt.plot(q, mae, "g--", label=r"MAE ($\ell_1$)")
    plt.plot(q, mse, "r--", linewidth=1, label=r"MSE ($\ell_2$)")
    plt.legend(loc="upper left", fontsize=14)
    plt.xlabel("Actual sparsity")
    plt.ylabel("Cost", rotation=0)
    plt.axis([0, 1, 0, 0.95])
    plt.grid(True)
    save_fig("sparsity_loss_plot")
    plt.show()


def plot_loss_history(history):
    plt.style.use("ggplot")
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    save_fig("sparse_ae_loss_vs_epoch_plot")
    plt.show()


def main(checkpoint_path, epochs):
    sparse_l1_ae = create_sparse_l1_autoencoder()
    sparse_l1_ae.compile(loss="mse", optimizer="nadam")

    # Build the model by running a forward pass
    sparse_l1_ae.build(input_shape=(None, 28, 28))
    sparse_l1_ae.summary()

    if os.path.exists(checkpoint_path):
        print("Loading model from checkpoint...")
        sparse_l1_ae.load_weights(checkpoint_path)
    else:
        history = train_autoencoder(sparse_l1_ae, X_train, X_valid, checkpoint_path, epochs)
        plot_loss_history(history)

    plot_reconstructions(sparse_l1_ae)
    save_fig("sparse_ae")
    plt.show()

    plot_sparsity_loss()


if __name__ == "__main__":
    from hiq import set_seed
    set_seed(has_tf=True)
    parser = argparse.ArgumentParser(description="Train and evaluate a sparse L1 autoencoder.")
    parser.add_argument("--checkpoint_path", type=str, default="sparse_l1_ae_mnist_fashion.h5",
                        help="Path to save/load the model checkpoint")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    args = parser.parse_args()

    try:
        main(args.checkpoint_path, args.epochs)
    except Exception as e:
        print(f"An error occurred: {e}")
