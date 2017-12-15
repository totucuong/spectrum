from unittest import TestCase
from spectrum.models.triple import Triple
from spectrum.inferences.multitruth import MultiTruth

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
    # def test_compute_accuracy(self):
    #     mt = MultiTruth()
    #     triples = list()
    #     triples.append(Triple("alice", "works_for", "ibm", 'fake.com'))
    #     triples.append(Triple('alice', 'works_for', 'ibm', 'official.com'))
    #     triples.append(Triple("alice", "works_for", "cisco", 'true.com'))
    #     triples.append(Triple("alex", "works_for", "oracle", 'xyz.com'))
    #     triples.append(Triple("alex", "works_for", "uber", 'affirmative.com', 0.3))
    #     triples.append(Triple("bobs", "works_for", "cisco", 'affirmative.com', 0.8))
    #     mt.fit(triples)
    #     self.assertAlmostEqual(0.55, mt.get_accuracy('affirmative.com'))

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

