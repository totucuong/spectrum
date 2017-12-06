from unittest import TestCase

__author__ = 'totucuong'
__date__ = '12/6/17'

DATA_HOME = '../data/'

from spectrum.datasets.book import Book
class TestBook(TestCase):

    def setUp(self):
        self.book = Book(DATA_HOME)

    def test_nclaims(self):
        assert self.book.nclaims == 33971

    def test_ntruths(self):
        assert self.book.ntruths == 100
