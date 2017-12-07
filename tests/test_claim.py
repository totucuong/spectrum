from unittest import TestCase

__author__ = 'totucuong'
__date__ = '12/5/17'


from spectrum.models.claim import Claim

class TestClaim(TestCase):
    def test_init(self):
        claim = Claim('cuong','bornin','hanoi')
        assert claim.predicate == 'bornin'

