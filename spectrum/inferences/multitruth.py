from spectrum.inferences.judge import Judge
from spectrum.models.triple import Triple
import collections
import math
import numpy as np

class MultiTruth(Judge):
    """
    This class implement our algorithm @cuong2017multitruth
    """
    def __init__(self):
        super().__init__()

    def fit(self, triples):
        super().index(triples)
        # probabilities
        self.entity_to_prior = [np.array([]) for i in range(self.nentities)]
        self.entity_to_marginal = [np.array([]) for i in range(self.nentities)]
        self.entity_to_likelihood = [np.array([]) for i in range(self.nentities)]

        self.__compute_prior()
        # self.__compute_source_accuracy()
        # self.__compute_likelihood()
        # self.__compute_marginal_likelihood()
        # self.__compute_posterior()
        # self.__compute_truth()

    def __compute_source_accuracy(self):
        pass

    def __compute_prior(self):
        """
        Compute prior of triples p(t). This prior is computed by average the confidence of all triples t that share
        the same t.fact, i.e., (subject,predicate,object)
        """
        print('computing prior...')
        for e in range(self.nentities):
            facts = self.entity_to_facts[e]
            fact_set = list(set(facts))
            conf_vec = self.entity_to_confs[e]
            prior = np.zeros(len(facts))
            for i in range(len(fact_set)):
                mask = facts == fact_set[i]
                prior[mask] = sum(conf_vec[mask])/len(conf_vec[mask])
            self.entity_to_prior[e] = prior
            print(self.entity[e], ' :', facts, self.entity_to_prior[e])

    def __compute_likelihood(self):
        pass

    def __compute_marginal_likelihood(self):
        pass

    def __compute_posterior(self):
        pass

    def __compute_truth(self):
        pass

    def get_prior(self, fact):
        """
        return prior belief of a fact, i.e., a triple

        Parameters
        ----------
        fact: a triple (subject,predicate,object)

        Returns
        -------
        prior probility of this fact if it exists or None otherwise
        """
        eidx = self.entityidx[fact.entity]
        result = self.entity_to_prior[eidx][fact.object == self.entity_to_facts[eidx]]
        if (len(result) == 0):
            return None
        return result[0]


    # def __init__(self):
    #     self.src_to_triples = collections.defaultdict(lambda: list())
    #     self.sources = set()
    #     self.accuracy = collections.defaultdict(lambda: 0)
    #     self.pred_to_triples = collections.defaultdict(lambda: list())
    #     self.degree = collections.defaultdict(lambda: 0)
    #
    #     # number of false value in the underlying domain of any entity (subject,predicate)
    #     self.n = 10
    #
    # def fit(self, triples):
    #     self.__index(triples)
    #     self.__compute_source_accuracies()
    #     self.__estimate_degrees()
    #
    # def __index(self, triples):
    #     for t in triples:
    #         self.src_to_triples[t.source].append(t)
    #         self.sources.add(t.source)
    #         self.pred_to_triples[t.predicate].append(t)
    #
    # def __compute_source_accuracies(self):
    #     for s in self.sources:
    #         self.accuracy[s] = self.__compute_accuracy(s)
    #
    # def __estimate_degrees(self):
    #     for pred in self.pred_to_triples:
    #         self.degree[pred] = self.learn_degree(self.pred_to_triples[pred])
    #
    #
    # def __get_triples(self, source):
    #     """
    #     return a list of triples that provided by a source
    #
    #     Parameters
    #     ----------
    #     source: a source
    #
    #     Returns
    #     -------
    #     a list of triples
    #     """
    #     return self.src_to_triples[source]
    #
    # def get_correct_triples(self):
    #     pass
    #
    # def learn_degree(self, triples):
    #     """
    #     Learn predicate degree of triples, degree(pred). We model the distribution of degree(pred) with Geom(theta).
    #     P(degree(pred) = k | theta) = (1-theta)^(k-1) * theta.
    #
    #     Let n be the number of entity (subject,predicate) that has predicate=pred.
    #
    #     [k1,...,kn] be a vector where ki is the number of triples of each entity (subject,predicate).
    #
    #     The maximum likelihood estimation of theta is theta_MLE = n/(k1 + ... + kn), where [k1, ..., kn].
    #
    #     The expected value of degree(pred) is 1/theta_MLE.
    #
    #     Parameters
    #     ----------
    #     triples: a list of triple about the same entity (subject,predicate)
    #
    #     Returns
    #     -------
    #     estimated degree of predicate
    #     """
    #
    #     facts = []
    #     fact_set = set()
    #     for t in triples:
    #         if t.fact not in fact_set:
    #             fact_set.add(t.fact)
    #             facts.append(t)
    #
    #     entity_count = collections.defaultdict(lambda: 0)
    #     for t in facts:
    #         entity_count[t.entity] = entity_count[t.entity] + 1
    #     theta_MLE =  len(entity_count.values()) / sum(entity_count.values())
    #     return math.ceil(1/theta_MLE)
    #
    # def __compute_accuracy(self, source):
    #     """
    #     Compute accuracy of source
    #
    #     Parameters
    #     ----------
    #     source: a source
    #
    #     Returns
    #     -------
    #     accuracy of the source
    #     """
    #     conf = []
    #     for t in self.__get_triples(source):
    #         conf.append(t.confidence)
    #     return sum(conf)/len(conf)
    #
    # def get_accuracy(self, source):
    #     return self.accuracy[source]
    #
    # def prob(self, t, is_true=True):
    #     """
    #     probablity of a triple t provided by t.source is true or false
    #
    #     Parameters
    #     ----------
    #     t: a triple
    #     """
    #     if is_true:
    #         self.accuracy[t.source] / self.degree[t.predicate]
    #     return (1 - self.accuracy[t.source]) / self.n
    #
    # def prior(self, t):
    #     """
    #     prior probability that a triple
    #     p(t) = average of confidence of t over all the sources that provide it
    #     """
    #     pass
    #
    #
    # def posterior(self, t):
    #     """
    #     posterior probability of a triple given data, where data is the list of triples that has (s,p) as its entity
    #     """
    #     pass
    #
    # def prob(self, data):
    #     """
    #     probability of data, where data is the list of triples that has (s,p) as its entity
    #     """