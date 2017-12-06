__author__ = 'totucuong'
__date__ = '12/5/17'

import collections

from spectrum.models.claim import Claim

class MajorityVote:
    """
    This class implement basic knowledge fusion using majority voting. That is, among conflicting claims, the claim that
    recive the highest votes from datasets sources will be considered as the correct claim.
    """

    def __init__(self):
        self.claims = collections.defaultdict(lambda: list())
        self.resolved_claims = collections.defaultdict(lambda: list())


    def fit(self, claims):
        """
        Resolve claims to true claims
        :param claims: list of claims
        :type claims: each claim is of type Claim
        :return: none
        :rtype: none
        """
        for c in claims:
            self.claims[c.subject+ '|' + c.predicate].append(c.object)

        self.__resolve()


    def __resolve(self):
        for sp in self.claims.keys():
            count = collections.defaultdict(lambda: 0)
            for o in self.claims[sp]:
                count[o] = count[o] + 1

            true_v = max(count, key=count.get)
            self.resolved_claims[sp].append(true_v)

    @property
    def truths(self):
        return self.resolved_claims






