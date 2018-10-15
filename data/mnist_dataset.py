import os
from keras.datasets import mnist
import tensorflow as tf

NUM_CLASSES = 10


def _example_dataset(x):
    x_dat = tf.data.Dataset.from_tensor_slices(x)
    x_dat = x_dat.map(tf.to_float, num_parallel_calls=11)
    x_dat = x_dat.map(lambda img: img / 255.0, num_parallel_calls=11)
    x_dat = x_dat.map(lambda img: tf.expand_dims(img, axis=-1), num_parallel_calls=11)

    return x_dat


def _target_dataset(y, classes=NUM_CLASSES):
    y_dat = tf.data.Dataset.from_tensor_slices(y)
    return y_dat.map(lambda x: tf.one_hot(x, classes), num_parallel_calls=11)


def create_datasets(path=f'{os.getcwd()}/data/mnist'):
    """
    Creates train and test MNIST tf.data.Dataset
    containing normalized images and one hot target vectors.
    Downloads data if necessary.


    `shapes: ((28, 28, 1), (10,)), types: (tf.float32, tf.float32)`

    Parameters
    ----------
    path: str
        path to dataset

    Returns
    -------
    (tf.data.Dataset, tf.data.Dataset)

    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data(path)

    x_dat = _example_dataset(x_train)
    y_dat = _target_dataset(y_train)
    train = tf.data.Dataset.zip((x_dat, y_dat))

    x_dat = _example_dataset(x_test)
    y_dat = _target_dataset(y_test)
    test = tf.data.Dataset.zip((x_dat, y_dat))

    return train, test
