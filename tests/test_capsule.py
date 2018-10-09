import tensorflow as tf
import numpy as np

import layers.capsule as caps
from layers import helpers


def createBatch(batch_size, capsules, caps_dim):
    return np.arange(batch_size * capsules * caps_dim).reshape((batch_size, capsules, caps_dim))


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


if __name__ == '__main__':
    tf.test.main()
