import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
import pandas as pd


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


def compute_truth(q_z):
    """compute truths

    Parameters
    ----------
    q_z: callable
        posterior distributions of hidden truth, z_truth_i

    Returns
    -------
    discovered_truths: pd.DataFrame
        A dataframe that contains inferred truths with columns ['object_id', 'value']
    """
    with ed.tape() as q_z_sample:
        q_z()
    object_id = []
    value = []
    for z_name, z_sample in q_z_sample.items():
        if z_name.startswith('z_truth'):
            object_id.append(int(z_name.split('_')[2]))
            value.append(z_sample.distribution.mode().numpy())
    return pd.DataFrame(data={'object_id': object_id, 'value': value})


def compute_trust_and_truth(q_z):
    """compute trusts and truths.
    
    Parameters
    ----------
    q_z: callable
        the variational approximation of posterior distributions of trusts
        and truths.

    Returns
    -------
    trust: dict
        a dictionary `{source_id, ed.RandomVariable}`. Some algorithmic
        truth discovery method such as majority voting or Truth Finder,
        does not model source reliability using distribution, instead they
        output a reliablity score. We capture this situation using
        ed.Deterministic(loc=reliablity_score). For other methods, such as
        LCAs, we use ed.Categorical to model reliablities of data sources.

    truth: dict
        a dictionary `{object_id, }` mapping `object_id`
        to an `ed.RandomVariable`. In spectrum, we model the uncertainty
        of truths using probability distribution, which is represented as
        a random variate `ed.RandomVariable`.
    """
    with ed.tape() as tape:
        q_z()
    trust = dict()
    trust['z_trusts'] = tape.pop('z_trusts')
    truth = tape.copy()
    return trust, truth
