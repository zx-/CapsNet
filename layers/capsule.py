import tensorflow as tf
from tensorflow import keras
from layers import helpers


def _coupling_logits(inputs, num_caps):
    """
    Creates coupling logits `b` with shape corresponding to input.
    Outputs `tf.ones` tensor with shape `(batch, num_caps[l], num_caps[l-1], 1)`

    Parameters
    ----------
    inputs: tf.Tensor
        Input tensor `(batch, num_caps[l-1], units[l-1])`
    num_caps: int
        Number of capsules in current layer `num_caps[l]`

    Returns
    -------
    tf.Tensor

    """
    with tf.name_scope("coupling_logits"):
        input_shape = tf.shape(inputs)
        batch = tf.gather(input_shape, 0)
        num_input_caps = tf.gather(input_shape, 1)
        return tf.ones([batch, num_caps, num_input_caps, 1])


def _coupling_coefficients(coupling_logits):
    """
    Takes input tensor of shape `(batch, num_caps[l], num_caps[l-1], 1)`.
    Applies softmax on axis `-2`.

    Parameters
    ----------
    coupling_logits: tf.Tensor
        Tensor with coupling logit values `b`.

    Returns
    -------
    tf.Tensor
        tensor of same shape as input

    """
    with tf.name_scope("coupling_coefficients"):
        return tf.nn.softmax(coupling_logits, axis=-2)


# def _matmul_over_caps(example, weights):
#     """
#     Takes one example of shape `(num_caps[l-1], 1, units[l-1])` and
#     weights of shape `(num_caps[l], num_caps[l-1], units[l-1], units[l])`.
#
#     Outputs prediction vectors for that example
#     with shape `(num_caps[l], num_caps[l-1], 1, units[l])`
#
#     Parameters
#     ----------
#     example: tf.Tensor
#         One input from lower level capsules.
#     weights: tf.Tensor
#         Weights used to compute predictions.
#
#     Returns
#     -------
#     tf.Tensor
#         Predictions u_hat for given example.
#     """
#     # return tf.map_fn(lambda x: _matmul_over_caps(x, weights), inputs)
#     return tf.map_fn(lambda caps_weights: tf.matmul(example, caps_weights), weights)
#

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
    with tf.name_scope("prediction_vectors"):
        # take inputs (batch,num_caps[l-1],units[l-1])
        # transform into (batch, 1, num_caps[l-1], 1, units[l-1])
        inputs = tf.expand_dims(inputs, -2)  # add dim before units
        inputs = tf.expand_dims(inputs, 1)  # add dim after batch

        # take weights (num_caps[l], num_caps[l-1], units[l-1], units[l])
        # transform into (1, num_caps[l], num_caps[l-1], units[l-1], units[l])
        weights = tf.expand_dims(weights, 0)

        # inputs (batch, num_caps[l], num_caps[l-1], 1, units[l-1])
        # weights (batch, num_caps[l], num_caps[l-1], units[l-1], units[l])
        inputs, weights = helpers.broadcast(inputs, weights, axis=[0, 1])

        return tf.matmul(inputs, weights)


def _routing(predictions, coupling_logits, routing_iterations):
    """
    Performs capsule routing.

    Parameters
    ----------
    predictions: tf.Tensor
        Lower level capsule predictions `(batch, num_caps[l], num_caps[l-1], 1, units[l])`
    coupling_logits: tf.Tensor
        Starting coupling logits `b` `(batch, num_caps[l], num_caps[l-1], 1)`
    routing_iterations: int
        Number of routing iterations

    Returns
    -------
    tf.Tensor
        Resulting capsule values. `(batch, num_caps[l], units[l])`

    """

    with tf.name_scope("routing"):
        for i in range(routing_iterations):
            coupling_coeffs = _coupling_coefficients(coupling_logits)
            out = tf.reduce_sum(tf.multiply(predictions, tf.expand_dims(coupling_coeffs, -1)), axis=-3)
            out = helpers.squash(out)  # (batch, num_caps[l], 1, units[l])
            if i < routing_iterations - 1:
                # expand to (batch, num_caps[l], 1, units[l], 1)
                out = tf.expand_dims(out, -1)
                # broadcast to (batch, num_caps[l], num_caps[l-1], units[l], 1)
                out, _ = helpers.broadcast(out, predictions, axis=-3, broadcast_b=False)
                # get logit update (batch, num_caps[l], num_caps[l-1], 1, 1)
                logits_update = tf.matmul(predictions, out)
                logits_update = tf.squeeze(logits_update, [-1])  # (batch, num_caps[l], num_caps[l-1], 1)
                # update coupling_logits
                coupling_logits = tf.add(coupling_logits, logits_update)

        return tf.squeeze(out, axis=-2)  # squeeze to get (batch, num_caps[l], units[l])


class Capsule(keras.layers.Layer):
    """
    Capsule layer.

    Computes predictions from input using trainable weights.
    Uses these predictions in routing to compute output capsules.

    Takes tensor of shape `(batch, input_caps, input_caps_dims)` as input.
    Outputs tensor of shape `(batch, capsules, capsule_units)`
    """

    def __init__(self,
                 capsules,
                 capsule_units,
                 routing_iterations=3,
                 create_weight_summary=False,
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
            'capsules': capsules,
            'capsule_units': capsule_units,
            'routing_iterations': routing_iterations,
            'create_weight_summary': create_weight_summary,
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
        if self.params['create_weight_summary']:
            helpers.variable_summaries(self.W, 'W')

        # if batch_size was fixed, we could create coupling_logits_b as tf.zeros too
        super(Capsule, self).build(input_shape)

    def call(self, inputs):
        coupling_logits_b = _coupling_logits(inputs, self.params['capsules'])
        predictions = _prediction_vectors(inputs, self.W)
        return _routing(predictions, coupling_logits_b, self.params['routing_iterations'])

    def compute_output_shape(self, input_shape):
        batch_size, _, _ = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([batch_size, self.params['capsules'], self.params['capsule_units']])
