from abc import abstractmethod
import collections

class Judge:
    """
    Abstract class to model a judge, who will decide which facts are correct/false
    """
    def __init__(self):
        self.claims = collections.defaultdict(lambda: list())
        self.resolved_claims = collections.defaultdict(lambda: list())

    @abstractmethod
    def fit(self, claims):
        pass

    @property
    def truths(self):
        return self.resolved_claims

