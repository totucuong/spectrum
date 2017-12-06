from unittest import TestCase

__author__ = 'totucuong'
__date__ = '12/5/17'

from spectrum.models.majority import MajorityVote
from spectrum.datasets.random import RandomData
from spectrum.datasets.population import Population

DATA_HOME = '../data/'
class TestMajorityVote(TestCase):
    def test_fit_random(self):
        claims = RandomData.generate()
        majority = MajorityVote()
        majority.fit(claims)
        resolves = majority.truths
        assert resolves['obama|born_in'][0] == 'usa'

    def test_fit_population(self):
        population = Population(DATA_HOME)
        majority = MajorityVote()
        majority.fit(population.claims)
        resolves = majority.truths
        assert resolves['abu dhabi|Population2006'][0] == 1000230
