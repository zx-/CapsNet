import tensorflow as tf
from tensorflow import keras
from layers import helpers


def _coupling_logits():
    pass


def _matmul_over_caps(example, weights):
    """
    Takes one example of shape `(num_caps[l-1], 1, units[l-1])` and
    weights of shape `(num_caps[l], num_caps[l-1], units[l-1], units[l])`.

    Outputs prediction vectors for that example
    with shape `(num_caps[l], num_caps[l-1], 1, units[l])`

    Parameters
    ----------
    example: tf.Tensor
        One input from lower level capsules.
    weights: tf.Tensor
        Weights used to compute predictions.

    Returns
    -------
    tf.Tensor
        Predictions u_hat for given example.
    """
    return tf.map_fn(lambda caps_weights: tf.matmul(example, caps_weights), weights)


def _prediction_vectors(inputs, weights):
    """
    Computes prediction vectors u_hat.

    Inputs should be of shape `(batch, num_caps[l-1], units[l-1])`.
    Weights should be of shape `(num_caps[l], num_caps[l-1], units[l-1], units[l])`

    Per each capsule[l] computes prediction vector for each lower level capsule.

    Outputs tensor of predictions `(batch, num_caps[l], num_caps[l-1], 1, units[l])`.

    Parameters
    ----------
    inputs: tf.Tensor
        Lower level capsules tensor.
    weights: tf.Tensor
        Weights used to compute predictions.

    Returns
    -------
    tf.Tensor
        Predictions u_hat.

    """
    # take inputs (batch,num_caps[l-1],units[l-1])
    # transform into (batch, num_caps[l-1], 1, units[l-1])
    inputs = tf.expand_dims(inputs, -2)  # add dim before units
    return tf.map_fn(lambda x: _matmul_over_caps(x, weights), inputs)


class Capsule(keras.layers.Layer):
    def __init__(self,
                 capsule_units,
                 capsules,
                 routing_iterations=3,
                 **kwargs):
        """

        Parameters
        ----------
        capsule_units: int
            Dimension of capsule vector
        capsules: int
            Number of capsules in layer
        routing_iterations: int
            Number of iterations done while routing.
        kwargs:
            passed to parent constructor
        """
        super(Capsule, self).__init__(**kwargs)
        self.params = {
            'capsule_units': capsule_units,
            'capsules': capsules,
            'routing_iterations': routing_iterations,
        }

    def build(self, input_shape):
        batch_size, num_input_caps, input_caps_dim = tf.TensorShape(input_shape).as_list()

        # weights of shape (capsules, input_caps, input_dims, output_dims)
        # first axis makes matrix multiplication easier
        # using this weights in matmul should output (batch,capsules,input_caps,1,output_dims)
        # this represents predictions from lower level capsules
        shape = tf.TensorShape([self.params['capsules'],
                                num_input_caps,
                                input_caps_dim,
                                self.params['capsule_units']])
        self.W = self.add_weight(name='kernel',
                                 shape=shape,
                                 initializer='uniform',
                                 trainable=True)
        # if batch_size was fixed, we could create coupling_logits_b as tf.zeros too
        super(Capsule, self).build(input_shape)

    def call(self, inputs):
        pass

    def compute_output_shape(self, input_shape):
        batch_size, _, _ = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([batch_size, self.params['capsules'], self.params['capsule_units']])
