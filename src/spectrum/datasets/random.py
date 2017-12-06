__author__ = 'totucuong'
__date__ = '12/5/17'

from spectrum.models.claim import Claim

class RandomData:
    """
    This class generate random dataset for easy testing.
    """

    @staticmethod
    def generate():
        """
        This method generate randomly a list of claims
        :return: a list of claims
        :rtype: Claim
        """
        claims = list()
        claims.append(Claim("obama", "born_in", "kenya"))
        claims.append(Claim("obama", "born_in", "usa"))
        claims.append(Claim("obama", "born_in", "indonesia"))
        claims.append(Claim("obama", "born_in", "usa"))
        return claims