from .truthdiscoverer import TruthDiscoverer


class LCA_EM(TruthDiscoverer):
    """Implement simpleLCA using EM

    References:
    - Latent Credibility Analysis, Jeff Pasternack and Dan Roth
    """
    def discover(self, claims, auxiliary_data=None, convergence_thres=1e-4):
        """Discover true claims and data source reliability

        Parameters
        ----------
        claims: pd.DataFrame
            a data frame that has columns `[source_id, object_id, value]`. We expect `source_id`, and
            `object_id` to of type `int`. `value` could be of type `int` if they are labels for things such as
            gender, diseases. It is of type `float` if it represents things such as sensor reading, etc.

        Returns
        -------
        trust: dict
            a dictionary `{source_id, ed.RandomVariable}`. Some algorithm-based truth discovery method such as majority voting
            or Truth Finder, does not model source reliability using distribution, instead they output a reliablity score. We
            capture this situation using ed.Deterministic(loc=reliablity_score). For other methods, such as LCAs, we use ed.Categorical
            to model reliablities of data sources.
            
        truth: dict
            a dictionary `{object_id, ed.RandomVariable}` mapping `object_id` to an `ed.RandomVariable`. In spectrum,
            we model the uncertainty of truths using probability distribution, which is represented as a random variate
            `ed.RandomVariable`.
        """
        self.claims = claims.copy()
        self._initialize()

    def _initialize(self):
        self.compute_observation_matrix()
        self.compute_weight_matrix()

    def compute_weight_matrix(self):
        """compute weight matrix W = [w_sm]that is is used to train

        s: index source. s is source_id
        m: index mutual execlusive set of claims, e.g. "Claimed Birth Years of Barack Obama". m is object_id.

        Returns
        -------
        W: np.ndarray
            a 2D matrix of shape (S,C) that represents observation. S is the number
            of data sources, C is the number of claims. 
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
        claim_train = self.claims[[
            'source_id', 'object_id', 'value', 'claim_id'
        ]].copy()
        B = claim_train.pivot(index='source_id',
                              columns='claim_id',
                              values='value')
        B.fillna(0, inplace=True)
        B[B > 0] = 1
        return B.values

    def _e_step(self):
        pass

    def _m_step(self):
        pass
