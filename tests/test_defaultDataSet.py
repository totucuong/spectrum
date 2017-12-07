from unittest import TestCase
from spectrum.datasets.dataset import DefaultDataSet

class TestDefaultDataSet(TestCase):

    def test_creating_default_dataset(self):
        dataset = DefaultDataSet('./data/claims.txt', './data/truths.txt')
        assert dataset.ntruths == 2
        assert dataset.nclaims == 5

