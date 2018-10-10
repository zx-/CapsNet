import tensorflow as tf
import numpy as np

import layers.capsule as caps
import layers.primary_caps as pcaps
from layers import helpers


def createBatch(batch_size, capsules, caps_dim):
    return np.arange(batch_size * capsules * caps_dim).reshape((batch_size, capsules, caps_dim))


def createBatchVolume(batch_size, h, w, ch):
    a = np.arange(h * w * ch).reshape((h, w, ch))
    return np.tile(a, (batch_size, 1, 1, 1)).astype(np.float32)


class PredictionVectorsTest(tf.test.TestCase):
    def prediction_vectors(self, a, b):
        with self.test_session():
            return caps._prediction_vectors(a, b).eval()

    def test_output_shape(self):
        a = createBatch(2, 3, 5).astype(float)
        b = np.random.rand(5, 3, 5, 7)
        output_shape = self.prediction_vectors(a, b).shape
        self.assertAllCloseAccordingToType(output_shape, [2, 5, 3, 1, 7])

    def test_output_values(self):
        np.random.seed(0)
        in1 = np.random.rand(5)
        in2 = np.random.rand(5)
        a = np.tile(np.array([in1, in2]), (2, 1, 1))  # (2,2,5)
        assert a.shape == (2, 2, 5)  # sanity check

        w1 = np.random.rand(5, 7)
        w2 = np.random.rand(5, 7)
        W = np.zeros((2, 2, 5, 7), float)
        W[0, 0, :, :] = w1
        W[0, 1, :, :] = w2
        W[1, 0, :, :] = w2
        W[1, 1, :, :] = w1

        o11 = np.matmul(in1, w1).reshape(1, 7)
        o12 = np.matmul(in1, w2).reshape(1, 7)
        o21 = np.matmul(in2, w1).reshape(1, 7)
        o22 = np.matmul(in2, w2).reshape(1, 7)
        # batches have same output we can omit first 2 in shape and tile it later
        output = np.zeros((2, 2, 1, 7), float)
        output[0, 0, :, :] = o11
        output[0, 1, :, :] = o22
        output[1, 0, :, :] = o12
        output[1, 1, :, :] = o21
        output = np.tile(output, (2, 1, 1, 1, 1))
        assert output.shape == (2, 2, 2, 1, 7)  # sanity check

        predicted_vectors = self.prediction_vectors(a, W)
        self.assertAllCloseAccordingToType(predicted_vectors, output)


class RoutingTest(tf.test.TestCase):
    def route(self, predictions, coupling_logits, routing_iterations):
        with self.test_session():
            return caps._routing(predictions, coupling_logits, routing_iterations).eval()

    def test_output_shape(self):
        pred = np.random.rand(5, 10, 15, 1, 8)
        coupling_logits = np.ones((5, 10, 15, 1))
        output = self.route(pred, coupling_logits, 3)

        self.assertAllCloseAccordingToType(output.shape, [5, 10, 8])

    def test_result_same(self):
        pred = np.ones((5, 3, 10, 1, 10))
        coupling_logits = np.ones((5, 3, 10, 1))

        expected = helpers.squash(np.ones((5, 3, 10)))

        output = self.route(pred, coupling_logits, 3)
        self.assertAllCloseAccordingToType(output, expected)

    def test_result_ones_value(self):
        pred = np.ones((5, 3, 10, 1, 10))
        coupling_logits = np.ones((5, 3, 10, 1))
        output = self.route(pred, coupling_logits, 3)
        self.assertAllInRange(output, 0.28747979 - 1e-06, 0.28747979 + 1e-06)

    def test_result_vector(self):
        pred = np.zeros((5, 3, 10, 1, 5))
        pred[:, :, 0, 0, :] = [1., 2., 4., 2., 1.]
        coupling_logits = np.ones((5, 3, 10, 1))
        output = self.route(pred, coupling_logits, 3)
        # [0.18885258, 0.37770516, 0.7554103 , 0.37770516, 0.18885258]
        exp_vec = [0.18669177, 0.37338353, 0.74676707, 0.37338353, 0.18669177]
        expected = np.tile(np.reshape(exp_vec, (1, 1, 5)), (5, 3, 1))
        self.assertAllCloseAccordingToType(output, expected)


class CapsuleTest(tf.test.TestCase):
    def infer(self, x, **kwargs):
        with self.test_session() as sess:
            x = tf.convert_to_tensor(x, dtype=float)
            layer = caps.Capsule(**kwargs)
            layer = layer(x)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            return layer.eval()

    def test_output_shape(self):
        input_caps = np.random.rand(50, 10, 5)  # batch, caps, units
        output = self.infer(input_caps, capsules=15, capsule_units=10)

        self.assertAllCloseAccordingToType(output.shape, [50, 15, 10])

    def test_combined_layers(self):
        with self.test_session() as sess:
            input_volume = createBatchVolume(10, 20, 20, 256)
            x = tf.convert_to_tensor(input_volume, dtype=float)
            primary_capsule = pcaps.PrimaryCaps(8, 32, 9)(x)
            capsule = caps.Capsule(10, 16)(primary_capsule)

            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            output = capsule.eval()
            self.assertAllCloseAccordingToType(output.shape, [10, 10, 16])


if __name__ == '__main__':
    tf.test.main()
