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
        claims.append(Claim("obama", "born_in", "kenya", source='fake.com'))
        claims.append(Claim("obama", "born_in", "usa", source='true.com'))
        claims.append(Claim("obama", "born_in", "indonesia", source='xyz.com'))
        claims.append(Claim("obama", "born_in", "usa", source='affirmative.com'))
        return claims