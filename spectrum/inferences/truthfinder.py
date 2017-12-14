from spectrum.inferences.judge import Judge
from spectrum.models.triple import Triple
import scipy.spatial.distance as dis
import numpy as np
from nltk.metrics import distance as sdis


class TruthFinder(Judge):
    """
    This class implement Truth Finder algorithm @yin2008truth
    """
    def __init__(self):
        super().__init__()

        # parameters set according to the paper
        self.rho = 0.5
        self.gamma = 0.3
        self.max_iter = 10
        self.sim_threshold = 0.1
        self.base_sim = 0

        # threshold of distance of trust score vector
        self.dis_tolerance = 0.1

        # confidence (probability of being correct)
        self.s = []

    def fit(self, triples):
        """
        Parameters:
        -----------
        claims:    a list of Claims

        Returns
        -------
        None
        """
        super().fit(triples)


        # trust score vector
        self.tau = -np.log(1 - np.ones(self.nsources) * 0.9)
        # trust score
        tau_dis = 99
        itr = 0
        while tau_dis > self.dis_tolerance and itr < self.max_iter:
            itr = itr + 1
            tau_old = np.copy(self.tau)
            self.s = self.__compute_conf()
            self.tau = self.__compute_trust()
            tau_dis = dis.cosine(tau_old, self.tau)
            print('iteration: %d\ttrust score distance: %f' %(itr, tau_dis))

    def __compute_conf(self):
        # new confidence
        s = []
        for i in range(self.nentities):
            # a set of claims (facts) about an entity i
            claim_set = list(set(self.entity_to_claims[i]))
            # confidence score of each claim
            sigma_i = np.zeros(len(claim_set))
            for j in range(len(claim_set)):
                srcs = self.entity_to_srcs[i]
                agree_srs = srcs[claim_set[j] == self.entity_to_claims[i]]
                sigma_i[j] = sum(self.tau[agree_srs])

            # compute adjusted confidence score
            s_i = np.zeros(len(self.entity_to_claims[i]))
            adj_sigma_i = np.copy(sigma_i)
            for j in range(len(claim_set)):
                implication = np.array([self.impl(claim_set[j], c) for c in claim_set])
                adj_sigma_i[j] = (1 - self.rho * implication[j]) * sigma_i[j] + \
                           self.rho * sum(sigma_i * implication)

                # compute confidence of claims about entity i
                s_i[self.entity_to_claims[i] == claim_set[j]] = 1/(1 + np.exp(-self.gamma*adj_sigma_i[j]))
            s.append(s_i)
        return s

    def impl(self, value1, value2):
        """
        implication function of (value1 -> value2)
        Truth Finder use impl(f1->f2) to measure how much fact f1 supports
        fact f2, where f1 and f2 is about the same entity.

        We assume now value1 and value2 are strings.
        """
        return np.exp(-sdis.edit_distance(value1,value2)) - self.base_sim

    def __compute_trust(self):
        """
        Compute new trust score
        """
        # trust
        t = np.zeros(self.nsources)

        # trust score
        tau = np.zeros(self.nsources)

        # number of facts provided per source
        source_size = np.zeros(self.nsources)

        # accumulate confidence of each entities into trust vector
        for e in range(self.nentities):
            t[self.entity_to_srcs[e]] = t[self.entity_to_srcs[e]] + self.s[e]
            source_size[self.entity_to_srcs[e]] = source_size[self.entity_to_srcs[e]] + 1

        # compute trust by averaging
        # t[source_size > 0] = t[source_size > 0] / source_size[source_size > 0]
        t = t / source_size
        tau = -np.log(1 - t)
        return tau

    def get_correct_triples(self):
        """
        Returns
        -------
        a list of correct triples
        """
        correct_triples = list()
        for e in range(self.nentities):
            true_claim = self.entity_to_claims[e][np.argmax(self.s[e])]
            sidxs = self.entity_to_srcs[e][true_claim == self.entity_to_claims[e]]
            sources = self.source[sidxs]
            splits = self.entity[e].split('|')
            correct_triples.append(Triple(splits[0], splits[1],true_claim, sources, np.max(self.s[e])))
        return correct_triples

    @property
    def sourcetrust(self):
        return self.tau

def main():
    from spectrum.models.triple import Triple
    truthfinder = TruthFinder()
    triples = list()
    triples.append(Triple("obama", "born_in", "kenya", 'fake.com'))
    triples.append(Triple('obama', 'born_in', 'kenya', 'official.com'))
    triples.append(Triple("obama", "born_in", "usa", 'true.com'))
    triples.append(Triple("obama", "born_in", "indonesia", 'xyz.com'))
    triples.append(Triple("obama", "born_in", "usa", 'affirmative.com'))
    triples.append(Triple('obama', 'profession', 'president', 'usa.com'))
    triples.append(Triple('obama', 'profession', 'lawyer', 'usa.com'))
    truthfinder.fit(triples)
    for t in truthfinder.get_correct_triples():
        print(t)

if __name__ == "__main__":
    main()