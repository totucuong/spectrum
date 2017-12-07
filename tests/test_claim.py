from unittest import TestCase
from spectrum.models.claim import Claim

class TestClaim(TestCase):
    def test_init(self):
        claim = Claim('cuong','bornin','hanoi')
        assert claim.predicate == 'bornin'

