import os
from keras.datasets import mnist
import tensorflow as tf

NUM_CLASSES = 10
NUM_PARALLEL = 11


def _example_dataset(x):
    x_dat = tf.data.Dataset.from_tensor_slices(x)
    x_dat = x_dat.map(tf.to_float, num_parallel_calls=NUM_PARALLEL)
    x_dat = x_dat.map(lambda img: img / 255.0, num_parallel_calls=NUM_PARALLEL)
    x_dat = x_dat.map(lambda img: tf.expand_dims(img, axis=-1), num_parallel_calls=NUM_PARALLEL)

    return x_dat


def _target_dataset(y, classes=NUM_CLASSES):
    y_dat = tf.data.Dataset.from_tensor_slices(y)
    return y_dat.map(lambda x: tf.one_hot(x, classes), num_parallel_calls=NUM_PARALLEL)


def create_datasets(path=f'{os.getcwd()}/data/mnist', make_dict=False):
    """
    Creates train and test MNIST tf.data.Dataset
    containing normalized images and one hot target vectors.
    Downloads data if necessary.


    `shapes: ((28, 28, 1), (10,)), types: (tf.float32, tf.float32)`

    Parameters
    ----------
    path: str
        path to dataset
    make_dict: bool
        make input image as part of dictionary

    Returns
    -------
    (tf.data.Dataset, tf.data.Dataset)

    """

    def img_to_dict(x): return {'image': x}

    (x_train, y_train), (x_test, y_test) = mnist.load_data(path)

    x_dat = _example_dataset(x_train)
    x_dat = x_dat.map(img_to_dict, NUM_PARALLEL) if make_dict else x_dat
    y_dat = _target_dataset(y_train)
    train = tf.data.Dataset.zip((x_dat, y_dat))

    x_dat = _example_dataset(x_test)
    x_dat = x_dat.map(img_to_dict, NUM_PARALLEL) if make_dict else x_dat
    y_dat = _target_dataset(y_test)
    test = tf.data.Dataset.zip((x_dat, y_dat))

    return train, test


def data_input_fn(dataset, batch_size=32, num_items=None, shuffle=False):
    """
    Returns data input function suitable for estimators.
    Data is served in batches from one_shot_iterator.

    Parameters
    ----------
    dataset: tf.data.Dataset
        Dataset to use.
    batch_size: int
        Batch size.
    num_items: int
        Takes first `num_items` items from dataset.
    shuffle: bool
        Whether to shuffle data before batching.

    Returns
    -------
    function
        input_fn
    """

    def input_fn():
        data = dataset

        if num_items is not None:
            data = data.take(num_items)

        if shuffle:
            data = data.shuffle(10000)

        data = data.batch(batch_size)
        data = data.prefetch(batch_size)

        return data.make_one_shot_iterator().get_next()

    return input_fn
