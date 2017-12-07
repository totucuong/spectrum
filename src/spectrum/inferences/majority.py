import collections

from spectrum.models.claim import Claim
from spectrum.inferences.judge import Judge

class MajorityVote(Judge):
    """
    This class implement basic knowledge fusion using majority voting. That is, among conflicting claims, the claim that
    recive the highest votes from datasets sources will be considered as the correct claim.
    """

    #
    #
    # def fit(self, claims):
    #     """
    #     Resolve claims to true claims
    #     """
    #     for c in claims:
    #         self.claims[c.subject+ '|' + c.predicate].append(c)
    #
    #     self.__resolve()
    def __init__(self):
        super().__init__()


    def fit(self, claims):
        # index claims
        for c in claims:
            self.claims[c.subject + '|' + c.predicate].append(c)

        # resolve claims
        for sp in self.claims.keys():
            count = collections.defaultdict(lambda: 0)
            invert = collections.defaultdict(lambda : list())
            for c in self.claims[sp]:
                count[c.object] = count[c.object] + 1
                invert[c.object].append(c)
            true_v = max(count, key=count.get)
            self.resolved_claims[sp].extend(invert[true_v])

        fitted = True

