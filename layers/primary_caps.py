import tensorflow as tf
from tensorflow import keras
from layers import helpers


class PrimaryCaps(keras.layers.Layer):
    """
    Layer with primary capsule functionality.

    Applies 2D convolution followed by reshape and squash activation.
    see `keras.layers.Conv2D`, `helpers.conv2caps` and `helpers.squash`

    Input should be a tensor of shape `(batch, height, width, channels)`.
    Outputs tensor of shape `(batch,num_of_capsules,conv_units)`
    """
    def __init__(self,
                 conv_units,
                 channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 **kwargs):
        """
        Parameters
        ----------
        conv_units: int
            Dimension of capsule vector
        channels: int
            Number of channels in convolutional capsule layer.
            Number of filters used in 2D convolution is `conv_units * channels`.
        kernel_size: int or (int,int)
            Same as in `keras.layers.Conv2D`
        strides: int or (int,int)
            Same as in `keras.layers.Conv2D`
        padding: string
            `valid` or `same` see `keras.layers.Conv2D`
        kwargs:
            passed to parent constructor
        """
        super(PrimaryCaps, self).__init__(**kwargs)
        self.params = {
            'conv_units': conv_units,
            'channels': channels,
            'kernel_size': kernel_size,
            'strides': strides,
            'padding': padding,
        }

    def build(self, input_shape):
        filters = self.params['channels'] * self.params['conv_units']

        # outputs (samples, new_rows, new_cols, filters)
        self.conv2d = keras.layers.Conv2D(
            filters,
            self.params['kernel_size'],
            strides=self.params['strides'],
            padding=self.params['padding'],
            name="conv2d"
        )

        super(PrimaryCaps, self).build(input_shape)

    def call(self, inputs):
        # apply convolution to input volume
        x = self.conv2d(inputs)
        # take (batch,h,w,ch) and reshape to (batch,num_of_capsules,conv_units)
        x = helpers.conv2caps(x, caps_dim=self.params['conv_units'])
        # apply squash to conv_units as activation function
        x = helpers.squash(x)
        return x

    def compute_output_shape(self, input_shape):
        batch_size, rows, cols, _ = tf.TensorShape(input_shape).as_list()

        stride_row, stride_col = helpers.unpack_tuple_int(self.params['strides'])

        if self.params['padding'].casefold() == 'same'.casefold():
            rows = helpers.conv_dim_same(rows, stride_row)
            cols = helpers.conv_dim_same(cols, stride_col)

        elif self.params['padding'].casefold() == 'valid'.casefold:
            kernel_row, kernel_col = self.__unpack_tuple_int(self.params['kernel_size'])
            rows = helpers.conv_dim_valid(rows, kernel_row, stride_row)
            cols = helpers.conv_dim_valid(cols, kernel_col, stride_col)

        else:
            raise ValueError(self.params['padding'] + ' is not valid padding')

        return tf.TensorShape([batch_size,
                               rows * cols * self.params['channels'],
                               self.params['conv_units']])
