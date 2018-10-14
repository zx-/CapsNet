import tensorflow as tf
import layers as caps_layers
import loss
import data
import numpy as np
import tensorflow.contrib.slim as slim
import layers.helpers as helpers
import os


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
        x = tf.multiply(capsules, tf.expand_dims(y_onehot, -1))
        x = tf.layers.flatten(x)
        x = tf.layers.Dense(units=512, activation='relu')(x)
        x = tf.layers.Dense(units=1024, activation='relu')(x)
        x = tf.layers.Dense(units=784, activation='relu')(x)
        x = tf.reshape(x, [-1, 28, 28, 1])

        return x


def build_graph(inputs, y_onehot):
    x = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=9,
                               padding='valid',
                               data_format='channels_last',
                               activation='relu')(inputs)

    x = caps_layers.PrimaryCaps(conv_units=8, channels=32, kernel_size=9, strides=2)(x)
    capsules = caps_layers.Capsule(capsules=10, capsule_units=16)(x)

    reconstruction = reconstruction_graph(capsules, y_onehot)

    return capsules, reconstruction


if __name__ == '__main__':
    RUN_NUM = 2
    NUM_EPOCHS = 5
    NUM_TRAIN = 10000
    NUM_TEST = 1000
    BATCH_SIZE = 32
    NUM_ITERATIONS = np.ceil(NUM_TRAIN / BATCH_SIZE)

    train, test = data.create_datasets()  # 60k, 10k

    train = train \
        .shuffle(20000) \
        .take(NUM_TRAIN) \
        .batch(BATCH_SIZE) \
        .repeat(NUM_EPOCHS)

    test = test \
        .shuffle(10000) \
        .take(NUM_TEST) \
        .batch(BATCH_SIZE) \
        .repeat(NUM_EPOCHS)

    train_it = train.make_one_shot_iterator()
    test_it = test.make_one_shot_iterator()

    train_X, train_y = train_it.get_next()
    test_X, test_y = train_it.get_next()

    capsules, reconstruction = build_graph(train_X, train_y)
    loss_fn = loss.total_loss(capsules, train_y, reconstruction, train_X)

    opt = tf.train.AdamOptimizer(0.03)
    training_operation = slim.learning.create_train_op(loss_fn, opt, summarize_gradients=True)

    # summaries
    tf.summary.scalar('total_loss', loss_fn)
    summaries = tf.summary.merge_all()
    helpers.print_model_summary()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(f'{os.getcwd()}/tmp/log/{RUN_NUM}', sess.graph)

        for i in range(2):
            summary, _ = sess.run([summaries, training_operation])
            writer.add_summary(summary, i)

        writer.flush()
        writer.close()
