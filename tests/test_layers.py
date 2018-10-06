import tensorflow as tf
import numpy as np
import unittest
from layers import layers


def createBatch(batch_size, h, w, ch):
    a = np.arange(h * w * ch).reshape((h, w, ch))
    return np.tile(a, (batch_size, 1, 1, 1))


class Conv2CapsTest(tf.test.TestCase):

    def infer(self, x, **kwargs):
        with self.test_session() as sess:
            x = tf.convert_to_tensor(x, dtype=float)
            p_caps = layers.PrimaryCaps(**kwargs)
            p_caps = p_caps(x)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            return p_caps.eval()

    def test_shape(self):
        self.assertAllEqual(
            self.infer(
                createBatch(3, 3, 3, 4),
                conv_units=2,
                channels=2,
                kernel_size=1
            ).shape,
            [3, 18, 2])

    def test_shape_2(self):
        self.assertAllEqual(
            self.infer(
                createBatch(3, 3, 3, 20),
                conv_units=2,
                channels=2,
                kernel_size=1
            ).shape,
            [3, 18, 2])

    def test_shape_3(self):
        self.assertAllEqual(
            self.infer(
                createBatch(10, 20, 20, 256),
                conv_units=8,
                channels=32,
                kernel_size=9,
                strides=2
            ).shape,
            [10, 1152, 8])
