from unittest import TestCase
from spectrum.datasets.book import Book
class TestBook(TestCase):

    def setUp(self):
        self.book = Book()

    def test_nclaims(self):
        assert self.book.nclaims == 33971

    def test_ntruths(self):
        assert self.book.ntruths == 100
