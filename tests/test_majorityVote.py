from unittest import TestCase

__author__ = 'totucuong'
__date__ = '12/5/17'

from spectrum.models.majority import MajorityVote
from spectrum.data.random import RandomData

class TestMajorityVote(TestCase):
    def test_fit(self):
        claims = RandomData.generate()
        majority = MajorityVote()
        majority.fit(claims)
        resolves = majority.get_resolved_claims()
        assert resolves['obama']['born_in'] == 'usa'



