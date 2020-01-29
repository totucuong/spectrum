from abc import ABC, abstractmethod


class TruthDiscoverer(ABC):
    """Truth discovery algorithm interface.
    """
    @abstractmethod
    def discover(self, claims):
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
        raise NotImplementedError()