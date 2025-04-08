from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_generator(latent_dim):
    """Build the generator model."""
    model = Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(alpha=0.01),
        BatchNormalization(momentum=0.8),
        Dense(256),
        LeakyReLU(alpha=0.01),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.01),
        BatchNormalization(momentum=0.8),
        Dense(3, activation='sigmoid')  # output layer for generating 3 features
    ])
    return model

def build_discriminator():
    """Build the discriminator model."""
    model = Sequential([
        Dense(512, input_shape=(3,)),
        LeakyReLU(alpha=0.01),
        Dense(256),
        LeakyReLU(alpha=0.01),
        Dense(128),
        LeakyReLU(alpha=0.01),
        Dense(1, activation='sigmoid')  # output: 1 neuron for real/fake classification
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def build_gan(generator, discriminator):
    """Build the combined GAN model."""
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model