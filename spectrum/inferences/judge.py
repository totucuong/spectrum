from abc import abstractmethod
import collections
import numpy as np

class Judge:
    """
    A knowledge fusion model, who will decide which facts are correct/false

    entity_to_claims

    entity_to_srcs
    """
    def __init__(self):
        self.entity = []
        self.entityidx = collections.defaultdict(lambda: 0)
        self.source = []
        self.sourceidx = collections.defaultdict(lambda: 0)

    def fit(self, triples):
        self.index(triples)

    def index(self, triples):
        seen_sources = set()
        seen_entities = set()
        for c in triples:
            if c.source not in seen_sources:
                seen_sources.add(c.source)
                self.source.append(c.source)
                self.sourceidx[c.source] = len(self.source) - 1

            if c.entity not in seen_entities:
                seen_entities.add(c.entity)
                self.entity.append(c.entity)
                self.entityidx[c.entity] = len(self.entity) - 1

        n = len(seen_entities)
        self.entity_to_facts = [list() for i in range(n)]
        self.entity_to_srcs = [list() for i in range(n)]
        self.entity_to_confs = [list() for i in range(n)]
        for c in triples:
            self.entity_to_facts[self.entityidx[c.entity]].append(c.object)
            self.entity_to_srcs[self.entityidx[c.entity]].append(self.sourceidx[c.source])
            self.entity_to_confs[self.entityidx[c.entity]].append(c.confidence)

        # convert to numpy array
        for i in range(self.nentities):
            self.entity_to_facts[i] = np.array(self.entity_to_facts[i])
            self.entity_to_srcs[i] = np.array(self.entity_to_srcs[i])
            self.entity_to_confs[i] = np.array(self.entity_to_confs[i])
        self.source = np.array(self.source)

    def get_entity_index(self, entity):
        return self.entityidx[entity]

    def get_claims(self, entity):
        return self.entity_to_facts[self.get_entity_index(entity)]

    def get_src_idx(self, source):
        return self.sourceidx[source]

    @property
    def nentities(self):
        return len(self.entity)

    @property
    def nsources(self):
        return len(self.source)

