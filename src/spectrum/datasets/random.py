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
        claims.append(Claim("obama", "born_in", "kenya", 'fake.com'))
        claims.append(Claim("obama", "born_in", "usa", 'true.com'))
        claims.append(Claim("obama", "born_in", "indonesia", 'xyz.com'))
        claims.append(Claim("obama", "born_in", "usa", 'affirmative.com'))
        return claims