from spectrum.inferences.judge import Judge
from spectrum.models.triple import Triple
import math
import numpy as np

class MultiTruth(Judge):
    """
    This class implement our algorithm @cuong2017multitruth
    """
    def __init__(self):
        super().__init__()
        # number of false value in the underlying domain of any entity (subject,predicate)
        self.nfalse = 10

    def fit(self, triples):
        super().index(triples)

        # degree of predicates
        self.degree = list()
        self.entity_to_predicate = np.ones(self.nentities, dtype=int)
        for e in range(self.nentities):
            self.entity_to_predicate[e] = self.predicateidx[self.entity[e].split('|')[1]]
        self.predicate_to_degree_vec = [list() for i in range(self.npredicates)]
        self.__compute_degree_vec()

        # probabilities
        self.entity_to_prior = [np.array([]) for i in range(self.nentities)]
        self.entity_to_marginal = np.zeros(self.nentities)
        self.entity_to_likelihood = [np.array([]) for i in range(self.nentities)]
        self.entity_to_posterior = [np.array([]) for i in range(self.nentities)]

        # source accuracy
        self.accuracy = np.ones(self.nsources)
        self.__estimate_degree_of_predicate()
        self.__compute_prior()
        self.__compute_source_accuracy()
        self.__compute_marginal_likelihood()
        self.__compute_likelihood()
        self.__compute_posterior()

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
        if (len(fact_set) < ntrue):
            # everything is considered to be true
            facts_mask = np.array([True] * len(facts))
            marginal = self.compute_scenario_likelihood(accuracy, facts_mask, degree, nfalse)
        else:
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

    def get_degree(self, predicate):
        """
        Get degree of a predicate

        Parameters
        ----------
        predicate: a predicate or its index

        Returns
        -------
        degree of the predicate
        """
        if isinstance(predicate, int):
            predidx = predicate
        else:
            predidx = self.predicateidx[predicate]
        return self.degree[predidx]

    def get_degree_vec(self, predicate):
        """
        Get degree vector of a predicate

        Parameters
        ----------
        predicate: a predicate or its index

        Returns
        -------
        the degree vector of the predicate, i.e, count of triples, each count is for one entity (subject,predicate)
        """
        if isinstance(predicate, int):
            predidx = predicate
        else:
            predidx = self.predicateidx[predicate]
        return self.predicate_to_degree_vec[predidx]

    def nchoosek_subset(self, n, k):
        """
        Generator of boolean vector of combination c(n,k): [0011] means pick the last 2 object from 4 objects.
        Implementation according to http://www.math.umbc.edu/~campbell/Computers/Python/probstat.html

        If combinations are thought of as binary vectors we can write them in order, so
        0011 < 0101 < 0110 < 1001 < 1010 < 1100. From this we see that the vectors whose top bit is zero are listed
        first, then those with top bit equal to one. The vectors whose top bit is zero has bottom three bits whose density is two and the vectors whose top bit is one have bottom three bits whose density is one.
        """
        if n < k:
            raise ValueError("k must be at most n")
        if n == k:
            yield [1] * n
        elif k == 0:
            yield [0] * n
        else:
            for leftlist in self.nchoosek_subset(n - 1, k):
                yield leftlist + [0]
            for leftlist in self.nchoosek_subset(n - 1, k - 1):
                yield leftlist + [1]

    def __compute_degree_vec(self):
        for e in range(self.nentities):
            pidx = self.predicateidx[self.entity[e].split('|')[1]]
            count = len(np.unique(self.entity_to_facts[e]))
            self.predicate_to_degree_vec[pidx].append(count)

    def __estimate_degree_of_predicate(self):
        """
        Estimate the degree of predicates
        """
        for p in range(self.npredicates):
            self.degree.append(self.__learn_degree(p))

    def __learn_degree(self, pidx):
        """
        Estimate degree of a predicate

        Parameters
        ----------
        pidx: index of a predicate

        Returns
        -------
        degree of the predicate
        """
        deg_vec = self.__degree_vec(pidx)
        return np.rint(np.average(deg_vec))

    def __degree_vec(self, pidx):
        """
        Parameters
        ----------
        pidx: index of a predicate

        Returns
        --------
        a numpy.array of count of triples index by entity.

        Example:
        triples = list()
        triples.append(Triple("alice", "works_for", "ibm", 'fake.com', 0.3))
        triples.append(Triple('alice', 'works_for', 'ibm', 'official.com', 0.2))
        triples.append(Triple("alice", "works_for", "cisco", 'fake.com', 0.3))
        triples.append(Triple("alex", "works_for", "oracle", 'affirmative.com', 0.4))
        triples.append(Triple("alex", "works_for", "uber", 'affirmative.com', 0.3))
        triples.append(Triple("bobs", "works_for", "cisco", 'affirmative.com', 0.8))
        triples.append(Triple("bobs", "works_for", "cisco", 'alternative.com', 0.2))

        For this set of triples, we have the degree vector of predicate 'works_for'
        [3 2 1] since alice has 3 triples, alex has 2 triples, and bobs has 1 triple.
        """
        return self.predicate_to_degree_vec[pidx]

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
            degree = self.degree[self.entity_to_predicate[e]]

            # compute correct term
            correct_term = np.zeros(len(facts))
            for i in range(len(correct_term)):
                correct_term[i] = np.prod(accuracy[facts == facts[i]] / degree)


            # compute reduced likelihood term
            reduced_likelihood = np.zeros(len(facts))
            for i in range(len(reduced_likelihood)):
                # compute P(De - {t})
                others = [facts[i] != other for other in facts]
                reduced_facts = facts[others]
                reduced_accuracy = accuracy[others]
                reduced_likelihood[i] = self.compute_expectation(reduced_facts, reduced_accuracy,
                                                                 degree -1,
                                                                 degree, self.nfalse)
            self.entity_to_likelihood[e] = correct_term * reduced_likelihood

    def __compute_marginal_likelihood(self):
        """
        Compute marginal likelihood P(De), where De is the list of triples about entity e=(subject,predicate)
        """
        print('Computing marginal likelihood P(De)...')
        for e in range(self.nentities):
            print('entity :', self.entity[e])
            facts = self.entity_to_facts[e]
            accuracy = self.accuracy[self.entity_to_srcs[e]]
            degree = self.degree[self.entity_to_predicate[e]]
            self.entity_to_marginal[e] = self.compute_expectation(facts, accuracy , degree,
                                                                  degree, self.nfalse)

    def __compute_posterior(self):
        """
        Compute posterior probability P(t|De) = P(De|t)P(t)/P(De)
        """
        print("Computing posterior P(t|De)...")
        for e in range(self.nentities):
            self.entity_to_posterior[e] = (self.entity_to_likelihood[e] * self.entity_to_prior[e]) / \
                                          self.entity_to_marginal[e]
            if self.entity_to_marginal[e] == 0:
                print(self.entity[e])
                raise ArithmeticError("Marginal is 0")

    def get_correct_triples(self):
        """
        Get the correct triples for all entities. For each entities, we select the top-k facts according to their
        posterior probabilities P(t|De), where k is the degree of the predicate of each entity.

        Returns
        -------
        a list of correct triples
        """
        correct_triples = list()
        for e in range(self.nentities):
            # determine subject, predicate
            splits = self.entity[e].split('|')
            subject = splits[0]
            predicate = splits[1]

            # determine the top k facts
            top_k = self.compute_top_k(self.degree[self.entity_to_predicate[e]] ,self.entity_to_facts[e],
                                       self.entity_to_posterior[e])
            top_k_facts = self.entity_to_facts[e][top_k]

            # their sources
            sources = self.source[self.entity_to_srcs[e][top_k]]

            triples = []
            for i in range(len(top_k_facts)):
                triples.append(Triple(subject, predicate, top_k_facts[i], sources[i]))
            correct_triples.extend(triples)
        return correct_triples

    def compute_top_k(self, k, fact, probability):
        """
        Compute top-k facts ranked by probability

        Parameters
        ----------
        k : the number of facts to be selected
        fact : a list of strings that represents fact
        probability : a list of probability of each fact, probabilit[i] is the probability of fact[i]

        Returns
        --------
        the indices of of top-k facts
        """
        if k > len(fact):
            raise ValueError('k must be at most length(fact)')
        if len(fact) != len(probability):
            raise ValueError('Every fact must have a probability')

        dtype = [('fact', 'U100'), ('probability', float), ('index', int)]
        index = range(len(fact))
        fact_with_conf = np.array([f for f in zip(fact, probability, index)], dtype=dtype)
        fact_with_conf = np.sort(fact_with_conf, order='probability')
        return [f[2] for f in fact_with_conf[-k:]]