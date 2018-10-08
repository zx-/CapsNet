import tensorflow as tf
import numpy as np

import layers.capsule as caps


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


if __name__ == '__main__':
    tf.test.main()
