from unittest import TestCase
from spectrum.datasets.random import RandomData
from spectrum.inferences.truthfinder import TruthFinder
from spectrum.models.claim import Claim

class TestTruthFinder(TestCase):
    def test_facts(self):
        # claims = RandomData.generate()
        claims = list()
        claims.append(Claim("obama", "born_in", "kenya", 'fake.com'))
        claims.append(Claim("obama", "born_in", "usa", 'true.com'))
        claims.append(Claim("obama", "born_in", "indonesia", 'xyz.com'))
        claims.append(Claim("obama", "born_in", "usa", 'affirmative.com'))
        claims.append(Claim('obama', 'profession', 'president', 'usa.com'))
        claims.append(Claim('obama', 'profession', 'lawyer', 'usa.com'))
        truth_finder = TruthFinder()
        truth_finder.fit(claims)
        assert truth_finder.claim[0][1] == 'usa'
        assert truth_finder.src[0][1] == 'true.com'

