from unittest import TestCase
from spectrum.datasets.random import RandomData

class TestRandomData(TestCase):

    def test_generate(self):
        claims = RandomData.generate()
        assert len(claims) == 4

