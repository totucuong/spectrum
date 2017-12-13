from unittest import TestCase
from spectrum.models.triple import Triple

class TestClaim(TestCase):
    def test_init(self):
        claim = Triple('cuong', 'bornin', 'hanoi')
        assert claim.predicate == 'bornin'

