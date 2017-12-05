__author__ = 'totucuong'
__date__ = '12/5/17'

from spectrum.models.claim import Claim

class RandomData:
    """
    This class generate random dataset for easy testing.
    """

    @staticmethod
    def generate():
        claims = set()
        claims.add(Claim("obama", "born_in", "kenya"))
        claims.add(Claim("obama", "born_in", "usa"))
        claims.add(Claim("obama", "born_in", "indonesia"))
        claims.add(Claim("obama", "born_in", "usa"))
        return claims