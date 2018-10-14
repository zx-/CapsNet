import tensorflow as tf


def margin_loss(prediction, target, m_positive=.9, m_negative=0.1, lambd=.5):
    """
    Computes margin loss.
    Target should have value `1` for class present
    represented by highly active capsule and `0` otherwise.

    Parameters
    ----------
    prediction: tf.Tensor
        Predictions of shape `(batch,num_caps,caps_dim)`
    target: tf.Tensor
        Targets of shape `(batch, num_caps)`
    m_positive: float
        length target for positive capsules
    m_negative: float
        length target for negative capsules
    lambd: float
        weight of negative capsules in loss

    Returns
    -------
    tf.Tensor
        Loss value
    """

    with tf.name_scope("margin_loss"):
        lengths = tf.norm(prediction, axis=-1)  # (batch, num_caps)
        positive = target * tf.maximum(0.0, m_positive - lengths) ** 2
        negative = (1 - target) * lambd * tf.maximum(0.0, lengths - m_negative) ** 2
        loss = positive + negative

    return tf.reduce_sum(loss)


def reconstruction_loss(prediction, target):
    """
    Computes sum of squared differences.
    Inputs should be of same shape.

    Parameters
    ----------
    prediction: tf.Tensor
    target: tf.Tensor

    Returns
    -------
    tf.Tensor
    """

    return tf.reduce_sum((prediction - target) ** 2)


def total_loss(prediction,
               target_class,
               reconstruction,
               target_image,
               reconstruction_weight=0.0005,
               m_positive=.9,
               m_negative=0.1,
               lambd=.5):
    """

    Combination of margin_loss and reconstruction_loss

    Parameters
    ----------
    prediction: tf.Tensor
        Predictions of shape `(batch,num_caps,caps_dim)`
    target_class: tf.Tensor
        Targets of shape `(batch, num_caps)`
    reconstruction: tf.Tensor
    target_image: tf.Tensor
    reconstruction_weight: float
    m_positive: float
    m_negative: float
    lambd: float

    Returns
    -------
    tf.Tensor

    """
    ml = margin_loss(prediction, target_class, m_positive, m_negative, lambd)
    rl = reconstruction_loss(reconstruction, target_image)
    return ml + reconstruction_weight * rl
