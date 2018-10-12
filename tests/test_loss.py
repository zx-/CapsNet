import tensorflow as tf
import numpy as np
import loss


class MarginLossTest(tf.test.TestCase):
    def test_sanity_check(self):
        with self.test_session():
            np.random.seed(1)
            prediction = np.random.rand(5, 10, 16).astype(np.float32)
            target = np.eye(10)[:5].astype(np.float32)

            output = loss.margin_loss(prediction, target)

            self.assertShapeEqual(np.array(1), output)
            self.assertAllGreater(output.eval(), 100)

    def test_true_vector(self):
        with self.test_session():
            prediction = np.zeros((1, 1, 10)).astype(np.float32)
            prediction[0, 0, 5] = 1

            target = np.ones([1, 1]).astype(np.float32)

            output = loss.margin_loss(prediction, target)

            self.assertAllEqual(output.eval(), 0.0)

    def test_false_vector(self):
        with self.test_session():
            prediction = np.zeros((1, 1, 10)).astype(np.float32)
            prediction[0, 0, 5] = 1

            target = np.zeros([1, 1]).astype(np.float32)

            output = loss.margin_loss(prediction, target)

            self.assertAllCloseAccordingToType(output.eval(), 0.405)

    def test_false_true_vector(self):
        with self.test_session():
            prediction = np.zeros((1, 2, 10)).astype(np.float32)
            prediction[0, 0, 5] = 0.5  # 0.16
            prediction[0, 1, 5] = 1  # 0.405 loss

            target = np.zeros([1, 2]).astype(np.float32)
            target[0, 0] = 1

            output = loss.margin_loss(prediction, target)

            self.assertAllCloseAccordingToType(output.eval(), 0.565)

    def test_batch(self):
        with self.test_session():
            prediction = np.zeros((1, 2, 10)).astype(np.float32)
            prediction[0, 0, 5] = 0.5  # 0.16
            prediction[0, 1, 5] = 1  # 0.405 loss

            target = np.zeros([1, 2]).astype(np.float32)
            target[0, 0] = 1

            prediction = np.tile(prediction, (2, 1, 1))
            target = np.tile(target, (2, 1))
            assert prediction.shape == (2, 2, 10)
            assert target.shape == (2, 2)

            output = loss.margin_loss(prediction, target)

            self.assertAllCloseAccordingToType(output.eval(), 0.565 * 2)


if __name__ == '__main__':
    tf.test.main()
