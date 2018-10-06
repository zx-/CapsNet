import tensorflow as tf
import numpy as np
import unittest
from layers import layers


class Conv2CapsTest(tf.test.TestCase):
    def setUp(self):
        # create (3,3,4) mat simulating conv volume
        self.x = np.arange(3 * 3 * 4).reshape((3, 3, 4))
        # create batch dim
        self.x = np.tile(self.x, (3, 1, 1, 1))

    def infer(self, x, **kwargs):
        with self.test_session():
            x = tf.convert_to_tensor(x, dtype=float)
            p_caps = layers.PrimaryCaps(**kwargs)
            # p_caps.build(tf.shape(x))
            p_caps = p_caps(x)
            return p_caps.eval()

    def test_shape(self):
        self.assertAllCloseAccordingToType(
            self.infer(
                self.x,
                conv_units=2,
                channels=2,
                kernel_size=1
            ).shape,
            [3, 2, 2])
