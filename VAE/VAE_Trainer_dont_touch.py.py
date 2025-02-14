import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
layers = tf.keras.layers
models = tf.keras.models
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
import os
import sys
import numpy as np

LATENT_DIM = 95
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
TRAIN_SIZE = 0.8
EPOCHS = 10
LOSS = 'mse'
DATA_DIR_PATH = 'VAE/dataset'
LOG_DIR = 'VAE/logs'
VAR_ENCODER_MODEL_PATH = 'VAE/var_auto_encoder_model'

@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.Model):

    def __init__(self,name = 'ENCODER',**kwargs):
        super().__init__(name = name ,**kwargs)

        self.latent_dim = LATENT_DIM

        self.conv1 = layers.Conv2D(32, (4, 4), activation='relu', strides=2, padding='same')
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(128, (4, 4), activation='relu', strides=2, padding='same')
        self.conv4 = layers.Conv2D(256, (3, 3), activation='relu', strides=2, padding='same')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1024, activation='relu')
        self.mu = layers.Dense(self.latent_dim)
        self.sigma = layers.Dense(self.latent_dim)

        self.N = tfp.distributions.Normal(loc=0.0, scale=1.0)
        self.kl = 0  

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense1(x)

        mu = self.mu(x)

        sigma = tf.exp(self.sigma(x))  
        z = mu + sigma * tfd.Normal(0.0, 1.0).sample(tf.shape(mu))  
        self.kl = tf.reduce_sum(sigma**2 + mu**2 - tf.math.log(sigma) - 0.5) 

        return z

    def save_model(self, path):

        self.save(path, save_format="tf")
        print(f"Encoder model saved at {path}")

    def get_config(self):
        config = super(Encoder, self).get_config()

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.Model):

    def __init__(self,name = 'DECODER',**kwargs):
        super().__init__(name = name,**kwargs)

        self.dense1 = layers.Dense(1024, activation='leaky_relu')
        self.dense2 = layers.Dense(160 * 80, activation='leaky_relu')

        self.unflatten = layers.Reshape((10, 5, 256))

        self.deconv1 = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='leaky_relu')
        self.deconv2 = layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='leaky_relu')
        self.deconv3 = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='leaky_relu')
        self.deconv4 = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')

    def call(self, x):
        if isinstance(x, tuple):
            x = x[0]

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.unflatten(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)

        return x

@tf.keras.utils.register_keras_serializable()
class VariationalAutoencoder(tf.keras.Model):

    def __init__(self,name='VARIATIONAL_AUTOENCODER',**kwargs):
        super().__init__(name = name,**kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder()


    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)



def data_processing():
    data_dir = DATA_DIR_PATH

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(160, 80),
        batch_size=BATCH_SIZE,
        class_mode="input",
        #subset='training'
    )

    test_data = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(160, 80),
        batch_size=BATCH_SIZE,
        class_mode="input"
    )

    train_size = int(TRAIN_SIZE * train_data.samples)
    val_size = train_data.samples - train_size

    train_steps_per_epoch = train_data.samples // train_data.batch_size
    val_steps_per_epoch = val_size // train_data.batch_size
    test_steps_per_epoch = test_data.samples // test_data.batch_size


    train_dataset = tf.data.Dataset.from_generator(lambda: train_data, output_types=(tf.float32, tf.float32), output_shapes=([None,160, 80, 3], [None,160, 80, 3])).take(train_size)
    val_dataset = tf.data.Dataset.from_generator(lambda: train_data, output_types=(tf.float32, tf.float32), output_shapes=([None,160, 80, 3], [None,160, 80, 3])).skip(train_size)
    test_dataset = tf.data.Dataset.from_generator(lambda: test_data, output_types=(tf.float32, tf.float32), output_shapes=([None,160, 80, 3], [None,160, 80, 3]))

    return train_dataset, test_dataset, train_steps_per_epoch, val_steps_per_epoch, test_steps_per_epoch

class LossHistory(tf.keras.callbacks.Callback):

    def __init__(self):
        super().__init__()

        self.log_dir = LOG_DIR
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        print()
        print(f"Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")

        with self.summary_writer.as_default():
            tf.summary.scalar('Training Loss', train_loss, step=epoch)
            tf.summary.scalar('Validation Loss', val_loss, step=epoch)


def main():

    conv_encoder = VariationalAutoencoder()

    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    conv_encoder.compile(optimizer = OPTIMIZER, loss = LOSS)

    train_data, test_data, train_steps_per_epoch, _, test_steps_per_epoch = data_processing()

    history_callback = LossHistory()

    conv_encoder.fit(
        train_data,
        steps_per_epoch = train_steps_per_epoch,
        epochs = EPOCHS,
        validation_data = test_data,
        validation_steps = test_steps_per_epoch,
        verbose = 1,
        callbacks = [history_callback]
    )

    conv_encoder.encoder.save_model(VAR_ENCODER_MODEL_PATH)

    encoder_model = tf.keras.models.load_model(VAR_ENCODER_MODEL_PATH)

    dummy_input = np.random.rand(1, 160, 80, 3).astype(np.float32)  


    output = encoder_model(dummy_input)
    print("Output shape:", output.shape)
    print(output)

    encoder_model.summary()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("\nTerminating...")