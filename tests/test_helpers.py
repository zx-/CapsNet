import tensorflow as tf
import numpy as np
import unittest
from layers import helpers


def expand(x):
    return np.tile(x, (3, 1))


def expand2(x):
    return np.tile(np.expand_dims(np.tile(x, (5, 1)), axis=0), (2, 1, 1))


class SquashTest(tf.test.TestCase):
    def setUp(self):
        self.single = np.array([0.0, 1.0, 0.0])
        self.single_out = np.array([0.0, 0.5, 0.0])
        self.single2 = np.array([0.0, 1.0, 15.0])
        self.single_out2 = np.array([0.0, 0.06622598, 0.99338963])

    def squash(self, x, expected_output):
        with self.test_session():
            s = helpers.squash(x)
            self.assertAllCloseAccordingToType(s.eval(), expected_output)

    def test_vec(self):
        self.squash(self.single, self.single_out)
        self.squash(self.single2, self.single_out2)

    def test_mul_vec(self):
        self.squash(expand(self.single), expand(self.single_out))
        self.squash(expand(self.single2), expand(self.single_out2))

    def test_mul_vec_2(self):
        self.squash(expand2(self.single), expand2(self.single_out))
        self.squash(expand2(self.single2), expand2(self.single_out2))


class Conv2CapsTest(tf.test.TestCase):
    def setUp(self):
        # create (3,3,4) mat simulating conv volume
        self.x = np.arange(3 * 3 * 4).reshape((3, 3, 4))
        # create batch dim
        self.x = np.tile(self.x, (3, 1, 1, 1))

        self.x_out = np.arange(3 * 3 * 4).reshape((1, -1, 2))
        self.x_out = np.tile(self.x_out, (3, 1, 1))

    def test_conv_2_caps(self):
        with self.test_session():
            s = helpers.conv2caps(self.x, 2)
            self.assertAllCloseAccordingToType(s.eval(), self.x_out)


class ConvDimValidTest(unittest.TestCase):

    def test_kernel(self):
        self.assertEqual(helpers.conv_dim_valid(28, 9, 1), 20)

    def test_stride(self):
        self.assertEqual(helpers.conv_dim_valid(20, 9, 2), 6)


class ConvDimSameTest(unittest.TestCase):

    def test_stride(self):
        self.assertEqual(helpers.conv_dim_same(28, 2), 14)


class BroadcastTest(tf.test.TestCase):
    def test_shape_simple(self):
        with self.test_session():
            s = helpers.broadcast(np.random.rand(10, 1, 15), np.random.rand(1, 15, 3))
            a, b = s[0].eval(), s[1].eval()
            self.assertAllCloseAccordingToType(a.shape, [10, 1, 15])
            self.assertAllCloseAccordingToType(b.shape, [10, 15, 3])

    def test_shape_2(self):
        with self.test_session():
            s = helpers.broadcast(
                tf.convert_to_tensor(np.random.rand(10, 1, 15, 1, 15)),
                tf.convert_to_tensor(np.random.rand(1, 10, 1, 15, 3)),
                axis=[0, 1],
                broadcast_b=False)
            a, b = s[0].eval(), s[1].eval()
            self.assertAllCloseAccordingToType(a.shape, [10, 10, 15, 1, 15])
            self.assertAllCloseAccordingToType(b.shape, [1, 10, 1, 15, 3])


if __name__ == '__main__':
    tf.test.main()
