import tensorflow as tf
import layers as caps_layers
import loss
import os
from keras.datasets import mnist
from layers.helpers import print_model_summary

import tensorflow.contrib.slim as slim

IMG_SIZE = 28
NUM_CLASSES = 10
EXAMPLE_NUM = 256


def simple_caps_net(inputs):
    x = tf.expand_dims(inputs, -1)
    pcaps = tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=9,
                                   padding='valid',
                                   data_format='channels_last',
                                   activation='relu')
    x = pcaps(x)

    x = caps_layers.PrimaryCaps(conv_units=8, channels=32, kernel_size=9, strides=2)(x)
    x = caps_layers.Capsule(capsules=10, capsule_units=16)(x)
    return x


def create_input_placeholders(img_size=IMG_SIZE):
    t_input_x = tf.placeholder(tf.float32, [None, img_size, img_size], name='x')
    t_input_y = tf.placeholder(tf.int32, [None], name='y')

    return t_input_x, t_input_y


def prepare_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data(f'{os.getcwd()}/data/mnist')
    x_train = x_train / 255.0

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':

    run_num = 1

    (x_train, y_train), (x_test, y_test) = prepare_data()
    p_x, p_target = create_input_placeholders()

    output = simple_caps_net(p_x)
    loss_fn = loss.margin_loss(output, tf.one_hot(p_target, NUM_CLASSES))

    opt = tf.train.AdamOptimizer(0.03)
    training_operation = slim.learning.create_train_op(loss_fn, opt, summarize_gradients=True)

    # summaries
    tf.summary.scalar('margin_loss', loss_fn)
    merged = tf.summary.merge_all()
    print_model_summary()

    session = tf.Session()

    writer = tf.summary.FileWriter(f'{os.getcwd()}/tmp/log/{run_num}', session.graph)

    session.run(tf.global_variables_initializer())

    for i in range(0, 20):
        summary, _ = session.run([merged, training_operation],
                                 feed_dict={p_x: x_train[:EXAMPLE_NUM, :, :], p_target: y_train[:EXAMPLE_NUM]})
        writer.add_summary(summary, i)

    writer.flush()
    writer.close()
    session.close()
