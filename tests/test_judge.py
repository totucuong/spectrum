from unittest import TestCase
from spectrum.models.triple import Triple
from spectrum.inferences.judge import Judge

class TestJudge(TestCase):
    def setUp(self):
        self.triples = list()
        self.triples.append(Triple("obama", "born_in", "kenya", 'fake.com'))
        self.triples.append(Triple('obama', 'born_in', 'kenya', 'official.com'))
        self.triples.append(Triple("obama", "born_in", "usa", 'true.com'))
        self.triples.append(Triple("obama", "born_in", "indonesia", 'xyz.com'))
        self.triples.append(Triple("obama", "born_in", "usa", 'affirmative.com'))
        self.triples.append(Triple('obama', 'profession', 'president', 'usa.com'))
        self.triples.append(Triple('obama', 'profession', 'lawyer', 'usa.com'))
        self.judge = Judge()
        self.judge.fit(self.triples)


    def test_nsources(self):
        self.assertEquals(self.judge.nsources, 6)

    def test_nentities(self):
        self.assertEquals(self.judge.nentities,2)

    def test_get_entity_index(self):
        self.assertEquals(self.judge.get_entity_index('obama|profession'), 1)

    def test_get_claims(self):
        self.assertEquals(len(self.judge.get_claims('obama|born_in')), 5)


