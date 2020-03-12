from tensorflow_probability import edward2 as ed
import tensorflow as tf


class BBVI:
    """Implement black-box variational inference using score function.

    References:
        1. Variational Bayesian Inference with Stochastic Search,
        John Paisley, David Blei, Michael Jordan.

    Parameters
    ----------
    model: callable
        a probabilistic model p(x,z|theta)

    variational_model: callable
        a variational probablistic model q(z|lambda)

    trainable_variables: list
        a list of tf.Variables that represents [theta, lambda].

    n_samples: int
        the number of samples to be used to estimate gradients of BBVI loss.
    """
    def __init__(self,
                 p,
                 q,
                 p_vars,
                 q_vars,
                 n_samples,
                 compute_variance=False,
                 n_gradient_sample=100):
        self.p = p
        self.q = q
        self.p_vars = p_vars
        self.q_vars = q_vars
        self.n_samples = n_samples
        self.train_loss = []
        self.compute_variance = compute_variance

    def train(self, epochs=1, learning_rate=1e-4):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta_1=0.9,
                                             beta_2=0.999)
        for epoch in range(epochs):
            loss, grads_and_vars = self._bbvi_step()
            self.train_loss.append(loss.numpy())
            if epoch % 100 == 0:
                print(f'iteration {epoch} -  loss {loss.numpy()}')
            optimizer.apply_gradients(grads_and_vars)

    def _bbvi_step(self):
        """compute a score loss and return its gradients based
        on the score function estimation.
        """
        log_p = ed.make_log_joint_fn(self.p)
        log_q = ed.make_log_joint_fn(self.q)

        p_log_prob = []
        q_log_prob = []

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.p_vars + self.q_vars)

            # z_s ~ q(z)
            q_z_samples = sample(self.n_samples, self.q)

            # replay z_s on p(x,z)
            for z_s in q_z_samples:
                q_log_prob.append(log_q(**z_s))
                with ed.tape() as model_sample:
                    with ed.interception(ed.make_value_setter(**z_s)):
                        self.p()
                p_log_prob.append(log_p(**model_sample))

            p_log_prob = tf.stack(p_log_prob)
            q_log_prob = tf.stack(q_log_prob)

            losses = p_log_prob - q_log_prob
            surrogate_loss = -tf.reduce_mean(
                q_log_prob * tf.stop_gradient(losses))
            loss = -(tf.reduce_mean(losses))

        q_grads = t.gradient(surrogate_loss, self.q_vars)
        p_grads = t.gradient(loss, self.p_vars)
        grads_and_vars = list(zip(q_grads, self.q_vars)) + list(
            zip(p_grads, self.p_vars))

        return loss, grads_and_vars


def sample(n_samples, model, *args, **kwargs):
    samples = []
    for i in range(n_samples):
        with ed.tape() as tape:
            model(*args, **kwargs)
        samples.append(tape)
    return samples


# def observe(observation, model, *args, **kwargs):
#     """compute observed model

#     Parameters
#     ----------
#     model: callable
#         a callable whose computation consists of with ed.RandomVariable's.

#     data: dict
#         a dictionary mapping ed.RandomVariable's name to its data.

#     Returns
#     -------
#     """
#     def observed_model():
#         with ed.interception(ed.make_value_setter(**observation)):
#             model(*args, **kwargs)

#     return observed_model
