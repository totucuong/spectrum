from unittest import TestCase
from spectrum.models.triple import Triple
from spectrum.inferences.multitruth import MultiTruth

class TestMultiTruth(TestCase):

    def test_learn_degree(self):
        mt = MultiTruth()
        triples = list()
        triples.append(Triple("alice", "works_for", "ibm", 'fake.com'))
        triples.append(Triple('alice', 'works_for', 'ibm', 'official.com'))
        triples.append(Triple("alice", "works_for", "cisco", 'true.com'))
        triples.append(Triple("alex", "works_for", "oracle", 'xyz.com'))
        triples.append(Triple("alex", "works_for", "uber", 'affirmative.com'))
        triples.append(Triple("bobs", "works_for", "cisco", 'affirmative.com'))
        deg = mt.learn_degree(triples)
        self.assertEqual(deg, 2)


