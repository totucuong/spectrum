from spectrum.inferences.judge import Judge
from spectrum.models.triple import Triple
import collections
import math

class MultiTruth(Judge):
    """
    This class implement our algorithm @cuong2017multitruth
    """

    def __init__(self):
        pass

    def fit(self, triples):
        pass

    def get_correct_triples(self):
        pass

    def learn_degree(self, triples):
        """
        Learn predicate degree of triples, degree(pred). We model the distribution of degree(pred) with Geom(theta).
        P(degree(pred) = k | theta) = (1-theta)^(k-1) * theta.

        Let n be the number of entity (subject,predicate) that has predicate=pred.

        [k1,...,kn] be a vector where ki is the number of triples of each entity (subject,predicate).

        The maximum likelihood estimation of theta is theta_MLE = n/(k1 + ... + kn), where [k1, ..., kn].

        The expected value of degree(pred) is 1/theta_MLE.

        Parameters
        ----------
        triples: a list of triple about the same entity (subject,predicate)

        Returns
        -------
        estimated degree of predicate
        """

        facts = []
        fact_set = set()
        for t in triples:
            if t.fact not in fact_set:
                fact_set.add(t.fact)
                facts.append(t)

        entity_count = collections.defaultdict(lambda: 0)
        for t in facts:
            entity_count[t.entity] = entity_count[t.entity] + 1
        theta_MLE =  len(entity_count.values()) / sum(entity_count.values())
        return math.ceil(1/theta_MLE)

    def compute_accuracy(self, source):
        """
        Compute accuracy of source

        Parameters
        ----------
        source: a source

        Returns
        -------
        accuracy of the source
        """
        pass
