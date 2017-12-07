from unittest import TestCase

__author__ = 'totucuong'
__date__ = '12/5/17'

from spectrum.inferences.majority import MajorityVote
from spectrum.datasets.random import RandomData
from spectrum.datasets.population import Population
from spectrum.datasets.book import Book

class TestMajorityVote(TestCase):
    def test_fit_random(self):
        claims = RandomData.generate()
        majority = MajorityVote()
        majority.fit(claims)
        resolves = majority.truths
        assert len(resolves['obama|born_in']) == 2

    def test_fit_population(self):
        population = Population()
        majority = MajorityVote()
        majority.fit(population.claims)
        resolves = majority.truths
        assert resolves['abu dhabi|Population2006'][0].object == 1000230

    def test_fit_book(self):
        book = Book()
        majority = MajorityVote()
        majority.fit(book.claims)
        resolves = majority.truths
        print(resolves)

