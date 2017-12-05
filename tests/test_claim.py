from unittest import TestCase

__author__ = 'totucuong'
__date__ = '12/5/17'


from spectrum.models.claim import Claim

class TestClaim(TestCase):
    def test_init(self):
        claim = Claim()
        assert claim.confidence == 1

    def test_set_subject(self):
        claim = Claim()
        claim.subject = 'albert_einstein'
        assert claim.subject == 'albert_einstein'

