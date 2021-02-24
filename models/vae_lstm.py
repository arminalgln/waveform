"""
Title: Variational AutoEncoder

"""

"""
## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
## Create a sampling layer
"""

#%%
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + z_var * epsilon


"""
## Build the encoder
"""
features =pd.read_pickle('data/fft_features_abs_clean.pkl')
x_train = abs(features.values)
number_of_samples = x_train.shape[1]
number_of_features = 7
number_of_timesteps = 500
x_train = x_train.transpose().reshape(number_of_samples,number_of_features,number_of_timesteps).transpose(0,2,1)

latent_dim = 3
#in_dim = 500*7
encoder_inputs = keras.Input(shape=(number_of_timesteps, number_of_features))
x = layers.LSTM(1)(encoder_inputs)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_var = layers.Dense(latent_dim, name="z_var")(x)
z = Sampling()([z_mean, z_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_var, z], name="encoder")
encoder.summary()


"""
## Build the decoder
"""

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(16, activation="relu")(latent_inputs)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
decoder_outputs = layers.Dense(number_of_timesteps*number_of_features, activation="relu")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


"""
## Define the VAE as a `Model` with a custom `train_step`
"""


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            mse = keras.losses.MeanSquaredError()
            number_of_samples = tf.shape(data)[0]
            print(number_of_samples)
            number_of_features = 7
            number_of_timesteps = 500
            data = tf.transpose(data, perm=[0,2,1])
            data = tf.reshape(data, [number_of_samples,number_of_features*number_of_timesteps])
            data = tf.expand_dims(data, -1)
            reconstruction_loss = mse(data, reconstruction)
            kl_loss = -0.5 * (1 + tf.math.log(tf.square(z_var)) - tf.square(z_mean) - tf.square(z_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
            grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


"""
## Train the VAE
"""



vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
#%%
vae.fit(x_train, epochs=10, batch_size=10)
#%%
causes = pd.read_pickle('data/causes.pkl')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
z_mean, _, _ = vae.encoder.predict(x_train)
scatter = ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2],c=list(causes['label']), cmap="Spectral")
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="causes")
ax.add_artist(legend1)
plt.show()

#%%
"""
## Display how the latent space clusters different digit classes
"""
labels = np.array(x_train.shape[0])

def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


plot_label_clusters(vae, x_train, labels)