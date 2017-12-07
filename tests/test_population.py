from unittest import TestCase
from spectrum.datasets.population import Population

class TestPopulation(TestCase):
    def setUp(self):
        self.population = Population()

    def test_nclaims(self):
        assert self.population.nclaims == 49955

    def test_ntruths(self):
        assert self.population.ntruths == 308

    def test_load_claims_correctly(self):
        assert self.population.claims[1].subject == "abu dhabi"
        assert self.population.claims[1].predicate == 'Population2006'
        assert self.population.claims[1].object == 1850230

    def test_load_truths_correctly(self):
        assert self.population.truths[1].subject == 'gary, indiana'
        assert self.population.truths[1].predicate == 'Population2000'
        assert self.population.truths[1].object == 102746