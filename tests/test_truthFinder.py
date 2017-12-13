from unittest import TestCase
from spectrum.datasets.random import RandomData
from spectrum.inferences.truthfinder import TruthFinder
from spectrum.models.triple import Triple

class TestTruthFinder(TestCase):

    # def test_truth(self):
    #     # claims = RandomData.generate()
    #     triples = list()
    #     triples.append(Triple("obama", "born_in", "kenya", 'fake.com'))
    #     triples.append(Triple('obama', 'born_in', 'kenya', 'official.com'))
    #     triples.append(Triple("obama", "born_in", "usa", 'true.com'))
    #     triples.append(Triple("obama", "born_in", "indonesia", 'xyz.com'))
    #     triples.append(Triple("obama", "born_in", "usa", 'affirmative.com'))
    #     triples.append(Triple('obama', 'profession', 'president', 'usa.com'))
    #     triples.append(Triple('obama', 'profession', 'lawyer', 'usa.com'))
    #     truth_finder = TruthFinder()
    #     truth_finder.fit(triples)

    def test_index(self):
        triples = list()
        triples.append(Triple("obama", "born_in", "kenya", 'fake.com'))
        triples.append(Triple('obama', 'born_in', 'kenya', 'official.com'))
        triples.append(Triple("obama", "born_in", "usa", 'true.com'))
        triples.append(Triple("obama", "born_in", "indonesia", 'xyz.com'))
        triples.append(Triple("obama", "born_in", "usa", 'affirmative.com'))
        triples.append(Triple('obama', 'profession', 'president', 'usa.com'))
        triples.append(Triple('obama', 'profession', 'lawyer', 'usa.com'))
        truth_finder = TruthFinder()
        truth_finder.fit(triples)
        # truth_finder


