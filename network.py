import tensorflow as tf
import functools
import numpy as np



def define_scope(function):
    # Decorator to lazy loading from https://danijar.com/structuring-your-tensorflow-models/
    attribute = '_cache_' + function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class VAE:

    def __init__(self, data, latent_dim, learning_rate, image_size=48, channels=3):
        self.data = data
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.inputs_decoder = ((image_size / 4)**2) * channels
        self.encode
        self.decode
        self.optimize


    @define_scope
    def encode(self):
        activation = tf.nn.relu
        with tf.variable_scope('Data'):
            x = self.data
        with tf.variable_scope('Encoder'):
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
            x = tf.layers.flatten(x)

            # Local latent variables
            self.mean_ = tf.layers.dense(x, units=self.latent_dim, name='mean')
            self.std_dev = tf.nn.softplus(tf.layers.dense(x, units=self.latent_dim), name='std_dev')  # softplus to force >0

            # Reparametrization trick
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.latent_dim]), name='epsilon')
            self.z = self.mean_ + tf.multiply(epsilon, self.std_dev)

            return self.z, self.mean_, self.std_dev

    @define_scope
    def decode(self):
        activation = tf.nn.relu
        with tf.variable_scope('Decoder'):
            x = tf.layers.dense(self.z, units=self.inputs_decoder, activation=activation)
            x = tf.layers.dense(x, units=self.inputs_decoder, activation=activation)
            recovered_size = int(np.sqrt(self.inputs_decoder/3))

            x = tf.reshape(x, [-1, recovered_size, recovered_size, 3])
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                           activation=activation)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                           activation=activation)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                           activation=activation)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                           activation=activation)

            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=48 * 48 * 3, activation=None)

            x = tf.layers.dense(x, units=48 * 48 * 3, activation=tf.nn.sigmoid)
            output = tf.reshape(x, shape=[-1, 48, 48, 3])
            output = tf.identity(output, name='decoded_output')

        return output

    @define_scope
    def optimize(self):
        with tf.variable_scope('Optimize'):
            # Reshape input and output to flat vectors
            flat_output = tf.reshape(self.decode, [-1, 48 * 48 * 3])
            flat_input = tf.reshape(self.data, [-1, 48 * 48 * 3])

            with tf.name_scope('loss'):
                img_loss = tf.reduce_sum(flat_input * -tf.log(flat_output) + (1 - flat_input) * -tf.log(1 - flat_output), 1)

                latent_loss = 0.5 * tf.reduce_sum(tf.square(self.mean_) + tf.square(self.std_dev) - tf.log(tf.square(self.std_dev)) - 1, 1)

                loss = tf.reduce_mean(img_loss + latent_loss)
                tf.summary.scalar('batch_loss', loss)

            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimizer

    def get_inputs_outputs(self):
        # Saving
        inputs = {
            'input_tensor': self.data,
        }
        outputs = {
            'latent': self.__getattribute__('_cache_encode')[0],
            'prediction': self.__getattribute__('_cache_decode'),
        }
        return inputs, outputs


