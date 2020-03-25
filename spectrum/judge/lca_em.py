from .truthdiscoverer import TruthDiscoverer
import numpy as np


class LCA_EM:
    """Implement simpleLCA using EM.

    Note: we should inherit from TruthDiscover after refactor it to 
    conform to new paradigmn:
        discoverer = TruthDiscover(claims)
        discoverer.discover()

    This implementation is specific to categorical claims.

    References:
    - Latent Credibility Analysis, Jeff Pasternack and Dan Roth

    Parameters
    ----------
    claims: pd.DataFrame
        a data frame that has columns `[source_id, object_id, value]`.
        We expect `source_id`, and `object_id` to of type `int`. `value` could
        be of type `int` if they are labels for things such as gender,
        diseases. It is of type `float` if it represents things such as sensor
        reading, etc.

    auxiliary_data: dict
        a dictionary that contain auxiliary data, e.g., source features.
        The default value is None.
    """
    def __init__(self, claims, auxiliary_data=None):
        # setup book keeping data structures
        self.claims = claims.copy()
        self.claims['claim_id'] = np.arange(0, len(claims))
        self.observation = self.compute_observation_matrix()
        self.weight = self.compute_weight_matrix()
        self.n_sources, self.n_objects, self.domain_size = self._compute_prob_desc(
        )
        # posterior p(o|X,\theta_old), o is m in the paper.
        self.posterior = dict()
        # prior p(o)
        self.prior = dict()
        for o in range(self.n_objects):
            self.prior[o] = np.ones(
                shape=(self.domain_size[o], )) / self.domain_size[o]

        # model parameters, source honesty
        self.theta_old = 0.5 * np.ones(shape=(self.n_sources, ))
        self.theta_new = 0.99 * np.ones(shape=(self.n_sources, ))

    def discover(self, alpha=1e-4):
        """Discover true claims and data source reliability

        Parameters
        ----------
        alpha: float
            convergence threshold for the EM algorithm. This is 
            L2 norm of (theta_old - theta_new), where theta is
            the simpleLCA parameters.

        Returns
        -------
        trust: dict
            a dictionary `{source_id, ed.RandomVariable}`.
            Some algorithm-based truth discovery method such as majority voting
            or Truth Finder, does not model source reliability using
            distribution, instead they output a reliablity score. We
            capture this situation using ed.Deterministic(loc=reliablity_score). 
            For other methods, such as LCAs, we use ed.Categorical to model
            reliablities of data sources.

        truth: dict
            a dictionary `{object_id, ed.RandomVariable}` mapping `object_id`
            to an `ed.RandomVariable`. In spectrum, we model the uncertainty
            of truths using probability distribution, which is represented as
            a random variate `ed.RandomVariable`.
        """
        while (np.linalg.norm(self.theta_old - self.theta_new, ord=2) > alpha):
            self._e_step()
            self._m_step()
        return self._compute_trust(), self._compute_truth()

    def _e_step(self):
        for o in range(self.n_objects):
            joint_prob = self._compute_joint(o)
            self.posterior[o] = joint_prob / np.sum(joint_prob)

    def _compute_joint(self, o_id):
        """compute p(o,B|theta_old)

        In the paper this is p(ym, X|theta_old).

        Parameters
        ----------
        o_id: int
            object id
        Returns
        -------
        p(o,B|theta_old): np.array
            joint distribution of object and data.
        """
        responsibility = np.ones(shape=(self.domain_size[o_id], ))
        for v in range(len(responsibility)):
            responsibility[v] = self.compute_responsibility(o_id, v)
        joint_prob = self.prior[o_id] * responsibility
        return joint_prob

    def compute_responsibility(self, o_id, v):
        """compute responsibility of data sources when making assertion.

        Responsibility of data sources with respect to value v on the domain
        of object o_id is defined as
            r = prod_{s}r_{s,v}

        where
            r_{s,v} = (theta_s**b_{s,v} * \prod_{c!=v}[(1-theta_s)/(|o|-1)]**b_{s,c})**w_so

        Parameters
        ----------
        o_id: int
            object id, i.e., m

        v: int
            a value on the object's domain

        Returns
        -------
        resp: float
            responsiblity of data sources with respect to the value v of
            object o_id.
        """
        resp = np.ones(shape=(self.n_sources, ))
        for s_id in range(resp):
            if self.weight[s_id][
                    o_id] == 1:  # weight=0 no need for computation
                if self.get_value(s_id, o_id) == v:
                    resp[s_id] = self.theta_old[s_id]
                else:
                    resp[s_id] = (1 - self.theta_old[s_id]) / (
                        self.domain_size[o_id])
        return np.prod(resp)

    def get_value(self, s_id, o_id):
        mask = (self.claims.source_id == s_id) & (
            self.claims.object_id == o_id)
        return self.claims[mask].value.values[0]

    def _m_step(self):
        pass

    def _compute_trust(self):
        return dict()

    def _comptue_truth(self):
        return dict()

    def compute_weight_matrix(self):
        """compute weight matrix weight = [w_so]that is is used to train

        s: index source. s is source_id
        o: index mutual execlusive set of claims, e.g. "Claimed Birth Years of Barack Obama". m is object_id.

        Returns
        -------
        weight: np.ndarray
            a 2D matrix of shape (S,O) that represents observation.
            S is the number of data sources, O is the number of objects
        """

        W = self.claims[['source_id', 'object_id',
                         'value']].pivot(index='source_id',
                                         columns='object_id',
                                         values='value')
        W.fillna(value=-1, inplace=True)
        W[W >= 0] = 1
        W[W < 0] = 0
        return W.values

    def compute_observation_matrix(self):
        """compute observation matrix B.

        B[source_id, claim_id] = value. When source source_id does not
        make claim claim_id then B[source_idd, claim_id] = -1.

        Returns
        -------
        B: 2d np.array
            observation matrix.
        """
        B = self.claims.pivot(index='source_id',
                              columns='claim_id',
                              values='value')
        B.fillna(-1, inplace=True)
        return B.values

    def _compute_prob_desc(self):
        """compute statistics of a given truth discovery problem.

        Returns
        -------
        n_sources: int
            number of data sources

        n_objects: int
            number of objects

        domain_size: dict
            domain_size[object_id] is the number of unique values of object
            object_id, i.e., |m| in the original paper.
        """
        problem_sizes = self.claims.nunique()
        n_sources = problem_sizes['source_id']
        n_objects = problem_sizes['object_id']
        domain_size = self.claims.groupby('object_id').max()['value'] + 1
        return n_sources, n_objects, domain_size
