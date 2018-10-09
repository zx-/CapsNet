import tensorflow as tf
import collections
import numpy as np


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


def broadcast(a, b, axis=0, broadcast_a=True, broadcast_b=True):
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

    Returns
    -------
    (tf.Tensor,tf.Tensor)
        a,b broadcasted

    """
    if not isinstance(axis, collections.Iterable):
        axis = [axis]

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
