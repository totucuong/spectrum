from tensorflow_probability import edward2 as ed
import tensorflow as tf


def print_trace(trace):
    for name, node in trace.nodes.items():
        if node['type'] == 'sample':
            print(f'{node["name"]} - sampled value {node["value"]}')


def observe(model, observation):
    """compute observed model

    Parameters
    ----------
    model: callable
        a callable whose computation consists of with ed.RandomVariable's.

    data: dict
        a dictionary mapping ed.RandomVariable's name to its data.

    Returns
    -------
    """
    def observed_model():
        with ed.interception(ed.make_value_setter(**observation)):
            model()

    return observed_model


def logits_for_uniform(batch_dim, domain_size):
    if domain_size > 1:
        return tf.math.log(tf.ones((batch_dim, domain_size)) / domain_size)
    return tf.math.log(0.5 * tf.ones((batch_dim, )))
