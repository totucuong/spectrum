__author__ = 'totucuong'
__date__ = '12/5/17'

import collections

from spectrum.models.claim import Claim

class MajorityVote:
    """
    This class implement basic knowledge fusion using majority voting. That is, among conflicting claims, the claim that
    recive the highest votes from data sources will be considered as the correct claim.
    """

    def __init__(self):
        self.claims = set()
        self.subjects = set()
        self.predicates = set()
        self.sub_claims = collections.defaultdict(lambda: collections.defaultdict(lambda: list()))
        self.resolved_claims = collections.defaultdict(lambda: collections.defaultdict(lambda: list()))


    def fit(self, claims):
        self.claims = set(claims)

        # sorting claims into (subject,predicate) subsets
        for c in claims:
            self.subjects.add(c.subject)
            self.predicates.add(c.predicate)
            self.sub_claims[c.subject][c.predicate].append(c)

        # resolve
        self.__resolve()


    def __resolve(self):
        for s in self.subjects:
            for p in self.predicates:
                count = collections.defaultdict(lambda: 0)
                for c in self.sub_claims[s][p]:
                    count[c.object] = count[c.object] + 1
                true_v = max(count, key=count.get)
                self.resolved_claims[s][p] = true_v


    def get_resolved_claims(self):
        return self.resolved_claims






