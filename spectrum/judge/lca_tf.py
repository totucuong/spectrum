import tensorflow as tf
from tensorflow_probability import edward2 as ed
from .truthdiscoverer import TruthDiscoverer


class LCA(TruthDiscoverer):
    def discover(self, claims, auxiliary_data=None):
        """Discover true claims and data source reliability

        Parameters
        ----------
        claims: pd.DataFrame
            a data frame that has columns `[source_id, object_id, value]`. We expect `source_id`, and
            `object_id` to of type `int`. `value` could be of type `int` if they are labels for things such as
            gender, diseases. It is of type `float` if it represents things such as sensor reading, etc.

        Returns
        -------
        truth: dict
            a dictionary `{object_id, ed.RandomVariable}` mapping `object_id` to an `ed.RandomVariable`. In spectrum,
            we model the uncertainty of truths using probability distribution, which is represented as a random variate
            `ed.RandomVariable`.

        trust: dict
            a dictionary `{source_id, ed.RandomVariable}`. Some algorithm-based truth discovery method such as majority voting
            or Truth Finder, does not model source reliability using distribution, instead they output a reliablity score. We
            capture this situation using ed.Deterministic(loc=reliablity_score). For other methods, such as LCAs, we use ed.Categorical
            to model reliablities of data sources.
        """
        self.n_sources, self.n_objects, self.domain_sizes = self._compute_prob_desc(
            claims)

    def _compute_prob_desc(self, claims):
        problem_sizes = claims.nunique()
        n_sources = problem_sizes['source_id']
        n_objects = problem_sizes['object_id']
        domain_sizes = claims.groupby('object_id').max()['value'] + 1
        return n_sources, n_objects, domain_sizes

    def model(self, claims):
        """a generative model

        We assume each source if it asserts an object's value then it is the
        one and the only assumption about that object made by it.

        Parameters
        ----------
        claims: pd.DataFrame
            a data frame that has columns [source_id, object_id, value]

        trainable_variables: list
            a list of tf.Variable. These are model parameters.
        """
        # hidden trusts

        honest_probs = tf.Variable(initial_value=tf.ones(self.n_sources) * 0.5,
                                   name='honest_probs')
        ed.Bernoulli(name=f'z_trusts', probs=honest_probs)

        # hidden truths
        object_probs = []
        z_truths = []
        for m in self.domain_sizes.index:
            object_probs.append(
                tf.Variable(initial_value=tf.ones(self.domain_sizes[m], ) /
                            self.domain_sizes[m],
                            name=f'truth_prob_{m}'))
            z_truths.append(
                ed.Categorical(name=f'z_truth_{m}', probs=object_probs[m]))

        # now generate claims
        x_claims = []
        for c in claims.index:
            s = claims.iloc[c]['source_id']
            m = claims.iloc[c]['object_id']
            z_truth_m = z_truths[m]
            probs = self._build_claim_probs(honest_probs[s],
                                            self.domain_sizes[m],
                                            z_truth_m.value)
            x_claims.append(ed.Categorical(name=f'x_claim_{c}', probs=probs))

        return [honest_probs] + object_probs

    def _build_claim_probs(self, honest_prob, domain_size, truth):
        mask = tf.reduce_sum(tf.one_hot([truth], domain_size), axis=0)
        other = tf.ones(domain_size) - mask
        probs = mask * honest_prob * tf.ones(domain_size) + other * (
            (1 - honest_prob) / (domain_size - 1)) * tf.ones(domain_size)

        return probs

    def mean_field_model(self, claims):
        """a mean field varational model
        Parameters
        ----------
        claims: pd.DataFrame
            a data frame that has columns [source_id, object_id, value]

        Returns
        -------
        trainable_variables: list
            a list of tf.Variable. These are variational model parameters.
        """
        # hidden trust
        honest_probs = tf.Variable(initial_value=tf.ones(self.n_sources) * 0.5,
                                   name='honest_probs_q')
        ed.Bernoulli(name=f'z_trusts', probs=honest_probs)

        # hidden truth
        object_probs = []
        for m in self.domain_sizes.index:
            object_probs.append(
                tf.Variable(initial_value=tf.ones(self.domain_sizes[m], ) /
                            self.domain_sizes[m],
                            name=f'truth_prob_{m}_q'))
            ed.Categorical(name=f'z_truth_{m}', probs=object_probs[m])

        return [honest_probs] + object_probs
