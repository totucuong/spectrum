import tensorflow as tf
import tensorflow_probability as tfp


def compute_gradient_variance(grad_samples):
    """compute variance of gradient estimation.

    Parameters
    ----------
    grad_samples: 2d array
        grad_samples[i,:] represents the ith gradient estimation.

    Returns
    -------
    grad_var: 1d array
        grad_var[j] represents the variance at the jth coordinate of
        a gradient estimation.
    """
    return tfp.stats.variance(grad_samples, sample_axis=0)


def flatten_gradients(grads_and_vars):
    """compute flatten gradients.
    """
    grads = []
    for grad, var in grads_and_vars:
        grads.append(tf.reshape(grad, shape=(-1, )))
    grads = tf.concat(grads, axis=0)
    return grads
