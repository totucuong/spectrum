from spectrum.models.triple import Triple

class RandomData:
    """
    This class generate random dataset for easy testing.
    """

    @staticmethod
    def generate():
        """
        This method generate randomly a list of claims
        :return: a list of claims
        :rtype: Triple
        """
        claims = list()
        claims.append(Triple("obama", "born_in", "kenya", 'fake.com'))
        claims.append(Triple("obama", "born_in", "usa", 'true.com'))
        claims.append(Triple("obama", "born_in", "indonesia", 'xyz.com'))
        claims.append(Triple("obama", "born_in", "usa", 'affirmative.com'))
        return claims