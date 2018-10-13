import tensorflow as tf
import collections
import numpy as np
import tensorflow.contrib.slim as slim

from distutils.version import LooseVersion


def squash(x, axis=-1):
    """
    Returns tensor of same shape with squash function applied
    along axis.

    Currently tested only on last axis [-1].

    Parameters
    ----------
    x : tf.Tensor
        input tensor with float type
    axis : int
        Default -1 as last axis.

    Returns
    -------
    tf.Tensor
        Squashed tensor.
    """
    with tf.name_scope("squash"):
        norm = tf.norm(x, axis=axis, keepdims=True)
        norm2 = norm ** 2
        output = norm2 / (norm2 + 1) * (x / norm)
        return output


def conv2caps(x, caps_dim):
    """
    Takes tensor of shape `(batch, h, w, channels)`
    and returns tensor of shape `(batch, None, caps_dim)`.

    channels should be divisible by caps_dim.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor
    caps_dim : int
        Number of units in capsule vector.

    Returns
    -------
    tf.Tensor

    """
    batch_size = tf.shape(x)[0]
    return tf.reshape(x, [batch_size, -1, caps_dim])


def conv_dim_valid(n, kernel, stride):
    """
    Computes dimension after convolution with padding `valid`.

    Parameters
    ----------
    n : int
        Input dimension when using convolution.
    kernel: int
        Kernel size in given dimension.
    stride : int
        Applied stride in given dimension.

    Returns
    -------
    int
       Dimension after convolution.

    """
    return np.ceil((n - kernel + 1) / stride)


def conv_dim_same(n, stride):
    """
    Computes dimension after convolution with padding `same`.

    Parameters
    ----------
    n : int
        Input dimension when using convolution.
    stride : int
        Applied stride in given dimension.

    Returns
    -------
    int
        Dimension after convolution.

    """
    return np.ceil(n / stride)


def unpack_tuple_int(val):
    """
    Returns `val` if iterable otherwise returns tuple `(val,val)`

    Parameters
    ----------
    val : collections.Iterable or int
        Single value or collection of values.

    Returns
    -------
    collections.Iterable
        Iterable with at least 2 values.

    """
    if isinstance(val, collections.Iterable):
        return val
    return val, val


def broadcast(a, b, axis=0, broadcast_a=True, broadcast_b=True, use_legacy=None):
    """
    Determines new dimensions as `a_shape[axis] = (max(a.shape[axis],b.shape[axis])`.
    Uses `tf.broadcast_to` method on `a` and `b` with new shape.
    Tensors should be of same rank.

    Parameters
    ----------
    a: tf.Tensor
        tensor to broadcast
    b: tf.Tensor
        tensor to broadcast
    axis: collections.Iterable or int
        axes on which to broadcast
    broadcast_a: bool
        if false broadcasting is not performed on a
    broadcast_b: bool
        if false broadcasting is not performed on b
    use_legacy: bool
        Use tf.broadcast_to instead of tf.tile
        defaults to tf.__version__ < 1.12.0

    Returns
    -------
    (tf.Tensor,tf.Tensor)
        a,b broadcasted

    """
    if not isinstance(axis, collections.Iterable):
        axis = [axis]

    if use_legacy is None:
        use_legacy = LooseVersion(tf.__version__) < LooseVersion('1.12.0')

    if not use_legacy:
        with tf.name_scope("two_tensor_broadcast"):
            a_shape = tf.unstack(tf.shape(a))
            b_shape = tf.unstack(tf.shape(b))
            new_dims = {}

            for ax in axis:
                new_dims[ax] = tf.maximum(a_shape[ax], b_shape[ax])

            for ax, dim in new_dims.items():
                a_shape[ax] = dim
                b_shape[ax] = dim

            if broadcast_a:
                a = tf.broadcast_to(a, a_shape)

            if broadcast_b:
                b = tf.broadcast_to(b, b_shape)

            return a, b

    else:
        with tf.name_scope("two_tensor_broadcast_legacy"):
            a_shape = tf.unstack(tf.shape(a))
            b_shape = tf.unstack(tf.shape(b))

            axis = list(map(lambda x: x % len(a_shape), axis))
            for i, (a_dim, b_dim) in enumerate(zip(a_shape, b_shape)):
                if i in axis:
                    new_dim = tf.maximum(a_dim, b_dim)
                    new_dim_a = tf.to_int32(new_dim / a_dim)
                    new_dim_b = tf.to_int32(new_dim / b_dim)
                else:
                    new_dim_a = 1
                    new_dim_b = 1

                a_shape[i] = new_dim_a
                b_shape[i] = new_dim_b

            if broadcast_a:
                a = tf.tile(a, a_shape)

            if broadcast_b:
                b = tf.tile(b, b_shape)

            return a, b


def variable_summaries(var, scope_name='summaries'):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(scope_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def print_model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
