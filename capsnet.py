import tensorflow as tf
import layers as caps_layers
import tensorflow.keras.layer as keras_layers
import loss


def reconstruction_graph(capsules, y_onehot):
    """

    Parameters
    ----------
    capsules: tf.Tensor
        `(batch, num_caps[l], units[l])`
    y_onehot: tf.Tensor
        `(batch, num_caps[l]`)

    Returns
    -------

    """

    with tf.name_scope('reconstruction'):
        x = capsules * y_onehot
        x = keras_layers.Dense(units=512, activation='relu')(x)
        x = keras_layers.Dense(units=1024, activation='relu')(x)
        x = keras_layers.Dense(units=784, activation='relu')(x)

        return x


def build_graph(inputs, y_onehot):
    x = keras_layers.Conv2D(filters=256,
                               kernel_size=9,
                               padding='valid',
                               data_format='channels_last',
                               activation='relu')(inputs)

    x = caps_layers.PrimaryCaps(conv_units=8, channels=32, kernel_size=9, strides=2)(x)
    capsules = caps_layers.Capsule(capsules=10, capsule_units=16)(x)

    reconstruction = reconstruction_graph(capsules, y_onehot)

    return capsules, reconstruction
