from unittest import TestCase

__author__ = 'totucuong'
__date__ = '12/5/17'

from spectrum.datasets.random import RandomData

class TestRandomData(TestCase):

    def test_generate(self):
        claims = RandomData.generate()
        assert len(claims) == 4

