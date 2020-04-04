import tensorflow as tf
from tensorflow_probability import edward2 as ed
from .utils import observe
from spectrum.inference.bbvi import BBVI
from spectrum.inference.utils import compute_trust_and_truth
import numpy as np


class LCA:
    """
    Parameters
        ----------
        claims: pd.DataFrame
            a data frame that has columns `[source_id, object_id, value]`. We
            expect `source_id`, and `object_id` to of type `int`. `value`
            could be of type `int` if they are labels for things such as
            gender, diseases. It is of type `float` if it represents things
            such as sensor reading, etc.

        auxiliary_data: dict
            a dictionary of auxliary data of some sort.
    """
    def __init__(self, claims, auxiliary_data=None):
        self.claims = claims.copy()
        self.auxiliary_data = auxiliary_data
        self._create_variables()
        self.observation = self._make_observation()
        self.observed_model = observe(self.model, self.observation)
        self.claim_to_s = dict()
        self.claim_to_m = dict()
        for c in range(self.claims.shape[0]):
            self.claim_to_s[c] = self.claims.iloc[c]['source_id']
            self.claim_to_m[c] = self.claims.iloc[c]['object_id']

    def discover(self,
                 epochs=1,
                 learning_rate=1e-4,
                 report_every=1,
                 n_samples=1,
                 compute_variance=False,
                 n_gradient_samples=5):
        """Discover true claims and data source reliability

        Parameters
        ----------
        n_samples: int
        the number of samples to be used to estimate gradients of BBVI loss.

        compute_variance: bool
            if compute_variance=False then variance of score-function gradient estimator
            is estimated at each epoch using n_gradient_samples.

        n_gradient_samples: bool
            the number of gradient estimation to be used when compute its variance. It will
            be ignored if compute_variance=False. 

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
            a dictionary `{object_id, ed.RandomVariable}` mapping `object_id`
            to an `ed.RandomVariable`. In spectrum, we model the uncertainty
            of truths using probability distribution, which is represented as
            a random variate `ed.RandomVariable`.
        """
        # peform black-box bvi
        self.bbvi = BBVI(p=self.observed_model,
                         q=self.mean_field_model,
                         p_vars=self.model_vars,
                         q_vars=self.latent_vars,
                         n_samples=n_samples,
                         compute_variance=compute_variance,
                         n_gradient_samples=n_gradient_samples)

        self.bbvi.train(epochs=epochs,
                        learning_rate=learning_rate,
                        report_every=report_every)

        return compute_trust_and_truth(self.mean_field_model)

    def _create_variables(self):
        """create trainable variables as well as other truth discovery parameters
        """
        print('initialize...')
        self.n_sources, self.n_objects, self.domain_sizes = self._compute_prob_desc(
            self.claims)

        print('number of sources: ', self.n_sources)
        print('number of objects: ', self.n_objects)

        # self.trainable_variables = []
        self.latent_vars = []
        self.model_vars = []

        print('initialize model parameters....')
        # model's parameter
        self.honest_logits_p = tf.Variable(initial_value=tf.random.uniform(
            shape=[self.n_sources], minval=-2, maxval=2),
                                           name='honest_logits_p')

        self._register(self.honest_logits_p, self.model_vars)

        self.object_logits_p = []
        for m in self.domain_sizes.index:
            self.object_logits_p.append(
                tf.Variable(initial_value=tf.random.uniform(
                    shape=[self.domain_sizes[m]], minval=-2, maxval=2),
                            name=f'truth_logits_{m}_p'))
        self._register(self.object_logits_p, self.model_vars)

        print('initialize guide parameters')
        # guide's parameter
        self.object_logits_q = []
        for m in self.domain_sizes.index:
            self.object_logits_q.append(
                tf.Variable(initial_value=tf.random.uniform(
                    shape=[self.domain_sizes[m]], minval=-2, maxval=2),
                            name=f'truth_logits_{m}_q'))
        self._register(self.object_logits_q, self.latent_vars)

    def _register(self, variable, collection):
        if isinstance(variable, list):
            collection.extend(variable)
        else:
            collection.append(variable)

    def _compute_prob_desc(self, claims):
        problem_sizes = claims.nunique()
        n_sources = problem_sizes['source_id']
        n_objects = problem_sizes['object_id']
        domain_sizes = claims.groupby('object_id').max()['value'] + 1
        return n_sources, n_objects, domain_sizes

    def _make_observation(self):
        """make observations
        """
        observation = dict()
        for c in self.claims.index:
            observation[f'x_claim_{c}'] = self.claims.iloc[c]['value']
        return observation

    def model(self):
        """a generative model

        We assume each source if it asserts an object's value then it is the
        one and the only assumption about that object made by it.
        """
        # p_truth
        z_truths = []
        for m in self.domain_sizes.index:
            z_truths.append(
                ed.Categorical(name=f'z_truth_{m}',
                               logits=tf.math.log_softmax(
                                   self.object_logits_p[m])))

        # claims
        x_claims = []
        for c in self.claims.index:
            s, m = self.get_s_m(c)
            z_truth_m = z_truths[m]
            honest_prob = tf.math.sigmoid(self.honest_logits_p[s])
            domain_size = self.domain_sizes[m]
            probs = self._build_claim_probs(honest_prob, domain_size,
                                            z_truth_m.value)
            x_claims.append(ed.Categorical(name=f'x_claim_{c}', probs=probs))

    def get_s_m(self, c_id):
        """return source and object ids"""
        return self.claim_to_s[c_id], self.claim_to_m[c_id]

    def _build_claim_probs(self, honest_prob, domain_size, truth):
        mask = tf.reduce_sum(tf.one_hot([truth], domain_size), axis=0)
        other = tf.ones(domain_size) - mask
        probs = mask * honest_prob * tf.ones(domain_size) + other * (
            (1 - honest_prob) / (domain_size - 1)) * tf.ones(domain_size)
        # honest_prob = honest_prob.numpy()
        # truth = truth.numpy()
        # probs = ((1 - honest_prob) / (domain_size - 1)) * np.ones(
        #     (domain_size, ))
        # probs[truth] = honest_prob
        return probs

    def mean_field_model(self):
        """a mean field varational model
        Parameters
        ----------
        claims: pd.DataFrame
            a data frame that has columns [source_id, object_id, value]
        """
        # q_truth
        for m in self.domain_sizes.index:
            ed.Categorical(name=f'z_truth_{m}',
                           logits=tf.math.log_softmax(self.object_logits_q[m]))