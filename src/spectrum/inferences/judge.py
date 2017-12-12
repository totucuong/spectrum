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

    def index(self, claims):
        self.__entity = []
        self.__entityidx = collections.defaultdict(lambda: 0)
        seen_entities = set()
        for c in claims:
            if c.entity not in seen_entities:
                seen_entities.add(c.entity)
                self.__entity.append(c.entity)
                self.__entityidx[c.entity] = len(self.__entity) - 1
        n = len(seen_entities)

        self.__claim = [list() for i in range(n)]
        self.__src = [list() for i in range(n)]
        for c in claims:
            self.__claim[self.__entityidx[c.entity]].append(c.object)
            self.__src[self.__entityidx[c.entity]].append(c.source)

    @property
    def claim(self):
        return self.__claim

    @property
    def src(self):
        return self.__src

    @property
    def entity(self):
        return self.__entity

    @property
    def entityindex(self):
        return self.__entityidx

