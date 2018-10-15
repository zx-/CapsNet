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
        x = tf.keras.layers.Dense(units=512, activation='relu')(x)
        x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
        x = tf.keras.layers.Dense(units=784, activation='sigmoid')(x)
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
    RUN = 'train_sigm_adam'
    NUM_EPOCHS = 5
    NUM_TRAIN = 10000
    NUM_TEST = 1000
    BATCH_SIZE = 64
    NUM_ITERATIONS = int(np.ceil(NUM_TRAIN / BATCH_SIZE))
    EVAL_ITERATIONS = int(np.ceil(NUM_TEST / BATCH_SIZE))

    # prepare inputs

    with tf.name_scope('datasets'):
        train, test = data.create_datasets()  # 60k, 10k
        train = train \
            .shuffle(20000) \
            .take(NUM_TRAIN) \
            .batch(BATCH_SIZE) \
            .repeat(NUM_EPOCHS) \
            .prefetch(BATCH_SIZE * 5)

        test = test \
            .shuffle(10000) \
            .take(NUM_TEST) \
            .batch(BATCH_SIZE) \
            .repeat(NUM_EPOCHS) \
            .prefetch(BATCH_SIZE * 5)

    iterator = tf.data.Iterator.from_structure(train.output_types, train.output_shapes)
    X, y = iterator.get_next()
    training_init_op = iterator.make_initializer(train)
    validation_init_op = iterator.make_initializer(test)

    # create graph
    capsules, reconstruction = build_graph(X, y)
    loss_fn, margin_loss, rec_loss = loss.total_loss(capsules, y, reconstruction, X)
    accuracy = loss.accuracy(capsules, y)

    opt = tf.train.AdamOptimizer(0.03)
    training_operation = slim.learning.create_train_op(loss_fn, opt, summarize_gradients=False)

    # summaries
    with tf.name_scope('train'):
        tf.summary.scalar('total_loss', loss_fn)
        tf.summary.scalar('margin_loss', margin_loss)
        tf.summary.scalar('rec_loss', rec_loss)
        tf.summary.scalar('acc', accuracy)
        summaries = tf.summary.merge_all()
    helpers.print_model_summary()

    # train & eval
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(f'{os.getcwd()}/tmp/log/{RUN}', sess.graph)

        for epoch in range(NUM_EPOCHS):

            sess.run(training_init_op)
            for i in range(NUM_ITERATIONS):
                summary, _ = sess.run([summaries, training_operation])
                writer.add_summary(summary, i + (epoch * NUM_ITERATIONS))

            sess.run(validation_init_op)
            eval_acc = []
            eval_loss = []
            for i in range(EVAL_ITERATIONS):
                summary_acc, summary_loss = sess.run([accuracy, loss_fn])
                eval_acc.append(summary_acc)
                eval_loss.append(summary_loss)

            summary = tf.Summary()
            summary.value.add(tag="eval/acc", simple_value=np.mean(eval_acc))
            writer.add_summary(summary, (epoch+1) * NUM_ITERATIONS)

            summary = tf.Summary()
            summary.value.add(tag="eval/loss", simple_value=np.mean(eval_loss))
            writer.add_summary(summary, (epoch + 1) * NUM_ITERATIONS)

        writer.flush()
        writer.close()
