from spectrum.inferences.judge import Judge
from spectrum.models.triple import Triple
import collections
import math
import numpy as np
from scipy.special import comb

class MultiTruth(Judge):
    """
    This class implement our algorithm @cuong2017multitruth
    """
    def __init__(self):
        super().__init__()
        # number of false value in the underlying domain of any entity (subject,predicate)
        self.nfalse = 10

        # default predicate degree (i.e., single-truth assumption
        self.default_degree = 1

    def fit(self, triples):
        super().index(triples)
        # probabilities
        self.entity_to_prior = [np.array([]) for i in range(self.nentities)]
        self.entity_to_marginal = np.zeros(self.nentities)
        self.entity_to_likelihood = [np.array([]) for i in range(self.nentities)]
        self.entity_to_posterior = [np.array([]) for i in range(self.nentities)]
        
        # source accuracy
        self.accuracy = np.ones(self.nsources)
        self.__compute_prior()
        self.__compute_source_accuracy()
        self.__compute_marginal_likelihood()
        self.__compute_likelihood()
        self.__compute_posterior()
        # self.__compute_truth()

    def compute_expectation(self, facts, accuracy, ntrue, degree, nfalse):
        """
        Compute marginal likelihood P(De)

        Parameters
        ----------
        facts: a list of facts
        accuracy: a list of source accuracy
        ntrue:  number of true facts within this list of facts
        degree: the degree of the predicate of the entity, e.g., degree of 'born_in'
        nfalse: the number of false values in the underlying domain of the predicate.

        Returns
        -------
        the expectation of P(facts)
        """
        if len(facts) == 0:
            return 1

        # set(De)
        fact_set = np.unique(facts)

        # marginal likelihood P(De)
        marginal = 0
        for mask in self.nchoosek_subset(len(fact_set), ntrue):
            fact_set_mask = np.array(mask) == 1
            selected_facts = fact_set[fact_set_mask]
            facts_mask = np.array([False] * len(facts))
            for i in range(len(facts)):
                if facts[i] in selected_facts:
                    facts_mask[i] = True
            marginal = marginal + self.compute_scenario_likelihood(accuracy, facts_mask, degree, nfalse)
        return marginal

    def compute_scenario_likelihood(self, accuracy, mask, pred_degree, nfalse):
        """
         Compute the likelihood of one scenario.

        A scenario is when given a list of facts. And we know which facts are correct, given by mask.
        We want to compute the probability of this scenario.

        Example:
            accuracy = [0.3, 0.4, 0.8]
            mask = [True, False, False]
            pred_degree = 2
            nfalse = 10
            The probability of this scenario is (0.3/2) * (1-0.4)/10 * (1 - 0.8)/10

        Parameters
        ----------
        accuracy: a list of source accuracy
        mask: a boolean np.array to determine which source provides true triple/false triple
        nfalse: the number false values in the underlying domain of predicate
        pred_degree: the degree of the predicate

        Returns
        -------
        conditional marginal likelihood
        """
        marginal_part = np.zeros(len(accuracy))
        marginal_part[mask] = accuracy[mask] / pred_degree
        marginal_part[~mask] = (1 - accuracy[~mask]) / nfalse
        return np.prod(marginal_part)


    def get_confidence(self, source):
        """
        Get the confidences of all triples provided by a source

        Parameters
        ----------
        source: a source or a source index

        Returns
        -------
        a np.array of confidences
        """
        if isinstance(source, int):
            src_idx = source
        else:
            src_idx = self.sourceidx[source]
        result = np.array([])
        for e in range(self.nentities):
            src_indices = self.entity_to_srcs[e]
            src_mask = src_indices == src_idx
            conf_vec = self.entity_to_confs[e][src_mask]
            if (len(conf_vec) != 0):
                result = np.concatenate((result, conf_vec))
        return result

    def get_accuracy(self, source):
        """
        Get the accuracy of a given source

        Parameters
        ----------
        source: the source index or its string value
        """
        if isinstance(source, int):
            src_idx = source
        else:
            src_idx = self.sourceidx[source]
        return self.accuracy[src_idx]

    def get_prior(self, fact):
        """
        Get prior belief of a fact, i.e., a triple

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

    def get_prior_of_entity(self, entity):
        """
        Get prior beliefs of an entity

        Parameters
        ----------
        entity: an entity or its index

        Returns
        -------
        An numpy.array of prior belief
        """
        if isinstance(entity, int):
            eidx = entity
        else:
            eidx = self.entityidx[entity]
        return self.entity_to_prior[eidx]

    def get_marginal_likelihood(self, entity):
        """
        Get the marginal likelihood P(De) of an entity. De is the set of triples that share the same
        entity=(subject,predicate)

        Parameters
        ----------
        entity: an entity index or its string representation, e.g., 'obama|born_in'
        """
        if isinstance(entity, int):
            eidx = entity
        else:
            eidx = self.entityidx[entity]
        return self.entity_to_marginal[eidx]

    def get_likelihood(self, entity, fact):
        """
        Parameters
        ----------
        entity: an entity or its index
        fact: a fact

        Returns
        -------
        P(De|fact)
        """
        if isinstance(entity, int):
            eidx = entity
        else:
            eidx = self.entityidx[entity]
        fidx = np.where(self.entity_to_facts[eidx] == fact)[0][0]
        return self.entity_to_likelihood[eidx][fidx]

    def get_likelihood_of_entity(self, entity):
        """
        Get likelihood of an entity

        Parameters
        ----------
        entity: an entity or its index

        Returns
        -------
        A numpy.array of likelihood of an entity
        """
        if isinstance(entity, int):
            eidx = entity
        else:
            eidx = self.entityidx[entity]
        return self.entity_to_likelihood[eidx]

    def get_posterior(self, entity):
        """
        Get posterior of an entity

        Parameters
        ----------
        entity: an entity of its index

        Returns
        -------
        A numpy.array of posterior of an entity
        """
        if isinstance(entity, int):
            eidx = entity
        else:
            eidx = self.entityidx[entity]
        return self.entity_to_posterior[eidx]


    def nchoosek_subset(self, n, k):
        """
        Generator of boolean vector of combination c(n,k): [0011] means pick the last 2 object from 4 objects.
        Implementation according to http://www.math.umbc.edu/~campbell/Computers/Python/probstat.html

        If combinations are thought of as binary vectors we can write them in order, so
        0011 < 0101 < 0110 < 1001 < 1010 < 1100. From this we see that the vectors whose top bit is zero are listed
        first, then those with top bit equal to one. The vectors whose top bit is zero has bottom three bits whose density is two and the vectors whose top bit is one have bottom three bits whose density is one.
        """
        if n == k:
            yield [1] * n
        elif k == 0:
            yield [0] * n
        else:
            for leftlist in self.nchoosek_subset(n - 1, k):
                yield leftlist + [0]
            for leftlist in self.nchoosek_subset(n - 1, k - 1):
                yield leftlist + [1]

    def __compute_source_accuracy(self):
        """
        Compute the accuracy of each source: A(p) = sum(conf)/len(conf), where conf is the confidence vector
        of all triples provided by the source p.
        """
        for s in range(self.nsources):
            conf_vec = self.get_confidence(s)
            self.accuracy[s] = np.average(conf_vec)

    def __compute_prior(self):
        """
        Compute prior of triples P(t). This prior is computed by average the confidence of all triples t that share
        the same t.fact, i.e., (subject,predicate,object)
        """
        print('Computing prior P(t)...')
        for e in range(self.nentities):
            facts = self.entity_to_facts[e]
            fact_set = list(set(facts))
            conf_vec = self.entity_to_confs[e]
            prior = np.zeros(len(facts))
            for i in range(len(fact_set)):
                mask = facts == fact_set[i]
                prior[mask] = sum(conf_vec[mask])/len(conf_vec[mask])
            self.entity_to_prior[e] = prior

    def __compute_likelihood(self):
        """
        Compute likelihood P(De|t) = [prod_over_rho_in_rho(t) of (A(rho)/k)] * P(De -{t}).

        We call the first term of P(De|t) as correct term.

        We call the second term of P(De|t) as reduced likelihood term.

        The term P(De-{t}) can be computed in a procedure similar to compute_marginal except that
        we need to select only k-1 triples every time instead of k, since t is already choosen to
        be correct.
        """
        print('Computing likelihood P(De|t)...')
        for e in range(self.nentities):
            facts = self.entity_to_facts[e]
            accuracy = self.accuracy[self.entity_to_srcs[e]]

            # compute correct term
            correct_term = np.zeros(len(facts))
            for i in range(len(correct_term)):
                #TODO: replace default degree with learnt degree
                correct_term[i] = np.prod(accuracy[facts == facts[i]] / self.default_degree)

            # compute reduced likelihood term
            reduced_likelihood = np.zeros(len(facts))
            for i in range(len(reduced_likelihood)):
                # compute P(De - {t})
                others = [facts[i] != other for other in facts]
                reduced_facts = facts[others]
                reduced_accuracy = accuracy[others]
                # TODO: when reduced_facts is empty then what is the expectation? =1 ? yes
                reduced_likelihood[i] = self.compute_expectation(reduced_facts, reduced_accuracy,
                                                                 self.default_degree -1,
                                                                 self.default_degree, self.nfalse)
            self.entity_to_likelihood[e] = correct_term * reduced_likelihood

    def __compute_marginal_likelihood(self):
        """
        Compute marginal likelihood P(De), where De is the list of triples about entity e=(subject,predicate)
        """
        print('Computing marginal likelihood P(De)...')
        for e in range(self.nentities):
            # TODO: for now we default degree = 1 (single truth assumption)
            facts = self.entity_to_facts[e]
            accuracy = self.accuracy[self.entity_to_srcs[e]]
            self.entity_to_marginal[e] = self.compute_expectation(facts, accuracy , self.default_degree,
                                                                  self.default_degree, self.nfalse)
    def __compute_posterior(self):
        """
        Compute posterior probability P(t|De) = P(De|t)P(t)/P(De)
        """
        print("Computing posterior P(t|De)...")
        for e in range(self.nentities):
            self.entity_to_posterior[e] = (self.entity_to_likelihood[e] * self.entity_to_prior[e]) / \
                                          self.entity_to_marginal[e]

    def __compute_truth(self):
        pass