from unittest import TestCase
from spectrum.models.triple import Triple
from spectrum.inferences.multitruth import MultiTruth
import numpy as np

class TestMultiTruth(TestCase):

    # def test_learn_degree(self):
    #     mt = MultiTruth()
    #     triples = list()
    #     triples.append(Triple("alice", "works_for", "ibm", 'fake.com'))
    #     triples.append(Triple('alice', 'works_for', 'ibm', 'official.com'))
    #     triples.append(Triple("alice", "works_for", "cisco", 'true.com'))
    #     triples.append(Triple("alex", "works_for", "oracle", 'xyz.com'))
    #     triples.append(Triple("alex", "works_for", "uber", 'affirmative.com'))
    #     triples.append(Triple("bobs", "works_for", "cisco", 'affirmative.com'))
    #     deg = mt.learn_degree(triples)
    #     self.assertEqual(deg, 2)
    #
    def test_get_accuracy(self):
        mt = MultiTruth()
        triples = list()
        triples.append(Triple("alice", "works_for", "ibm", 'fake.com'))
        triples.append(Triple('alice', 'works_for', 'ibm', 'official.com'))
        triples.append(Triple("alice", "works_for", "cisco", 'true.com'))
        triples.append(Triple("alex", "works_for", "oracle", 'xyz.com'))
        triples.append(Triple("alex", "works_for", "uber", 'affirmative.com', 0.3))
        triples.append(Triple("bobs", "works_for", "cisco", 'affirmative.com', 0.8))
        mt.fit(triples)
        self.assertAlmostEqual(0.55, mt.get_accuracy('affirmative.com'))

    def test_get_prior(self):
        mt = MultiTruth()
        triples = list()
        triples.append(Triple("alice", "works_for", "ibm", 'fake.com', 0.3))
        triples.append(Triple('alice', 'works_for', 'ibm', 'official.com', 0.2))
        triples.append(Triple("alice", "works_for", "cisco", 'true.com', 0.3))
        triples.append(Triple("alex", "works_for", "oracle", 'xyz.com', 0.4))
        triples.append(Triple("alex", "works_for", "uber", 'affirmative.com', 0.3))
        triples.append(Triple("bobs", "works_for", "cisco", 'affirmative.com', 0.8))
        triples.append(Triple("bobs", "works_for", "cisco", 'alternative.com', 0.2))
        mt.fit(triples)
        prior = mt.get_prior(Triple('alice','works_for','ibm'))
        self.assertAlmostEqual(prior, 0.25)

    def test_get_confidence(self):
        mt = MultiTruth()
        triples = list()
        triples.append(Triple("alice", "works_for", "ibm", 'fake.com', 0.3))
        triples.append(Triple('alice', 'works_for', 'ibm', 'official.com', 0.2))
        triples.append(Triple("alice", "works_for", "cisco", 'fake.com', 0.3))
        triples.append(Triple("alex", "works_for", "oracle", 'affirmative.com', 0.4))
        triples.append(Triple("alex", "works_for", "uber", 'affirmative.com', 0.3))
        triples.append(Triple("bobs", "works_for", "cisco", 'affirmative.com', 0.8))
        triples.append(Triple("bobs", "works_for", "cisco", 'alternative.com', 0.2))
        mt.fit(triples)
        self.assertTrue(np.all(mt.get_confidence('affirmative.com') == np.array([0.4, 0.3,0.8])))
        self.assertTrue(np.all(mt.get_confidence('fake.com') == np.array([0.3, 0.3])))

    def test_nchoosk_subset(self):
        mt = MultiTruth()
        comb = []
        for c in mt.nchoosek_subset(4, 2):
            comb.append(np.array(c))
        print(comb[5])
        self.assertTrue(np.all(np.array([0, 0, 1, 1]) == comb[5]))
        self.assertTrue(np.all(np.array([1,1,0,0]) == comb[0]))

    def test_compute_marginal_part(self):
        mt = MultiTruth()
        degree = 2
        nfalse = 5
        accuracy = np.array([0.4, 0.2, 0.6, 0.8])
        mask = np.array([True, False, False, True])
        marginal_part = mt.compute_marginal_part(accuracy, mask, degree, nfalse)
        print(marginal_part)
        self.assertAlmostEqual(marginal_part, 0.001024)

    def test_compute_marginal(self):
        mt = MultiTruth()
        triples = list()
        triples.append(Triple("alice", "works_for", "ibm", 'fake.com', 0.3))
        triples.append(Triple('alice', 'works_for', 'ibm', 'official.com', 0.2))
        triples.append(Triple("alice", "works_for", "cisco", 'fake.com', 0.3))
        triples.append(Triple("alex", "works_for", "oracle", 'affirmative.com', 0.4))
        triples.append(Triple("alex", "works_for", "uber", 'affirmative.com', 0.3))
        triples.append(Triple("bobs", "works_for", "cisco", 'affirmative.com', 0.8))
        triples.append(Triple("bobs", "works_for", "cisco", 'alternative.com', 0.2))
        mt.fit(triples)
        marginal = mt.compute_marginal('alice|works_for', 1, 10)
        self.assertAlmostEqual(marginal,0.00588)

