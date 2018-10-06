import tensorflow as tf
from tensorflow import keras
from layers import helpers


class PrimaryCaps(keras.layers.Layer):
    def __init__(self,
                 conv_units,
                 channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 **kwargs):
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
            padding=self.params['padding']
        )

        super(PrimaryCaps, self).build(input_shape)

    def call(self, inputs):
        x = self.conv2d(inputs)
        x = helpers.conv2caps(x, caps_dim=self.params['conv_units'])
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
