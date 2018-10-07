import tensorflow as tf
from tensorflow import keras
from layers import helpers


def _create_coupling_logits():
    pass

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
        pass

    def call(self, inputs):
        pass

    def compute_output_shape(self, input_shape):
        pass
