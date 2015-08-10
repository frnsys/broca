from broca import Pipe
from broca.common.shared import spacy
from broca.common.util import parallel


class Entities(Pipe):
    """
    Extracts entities and uses them as tokens.
    """
    input = Pipe.type.docs
    output = Pipe.type.entities

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def __call__(self, docs):
        if self.n_jobs == 1:
            return [self._extract(doc) for doc in docs]
        else:
            return parallel(self._extract, docs, self.n_jobs)

    def _extract(self, doc):
        res = spacy(doc, entity=True, tag=True, parse=False)
        return [Entity(e.string, e.label_) for e in res.ents]


class Entity():
    def __init__(self, name, label):
        self.name = name.strip().lower()
        self.label = label

    def __eq__(self, other):
        return self.name == other.name and self.label == other.label

    def __hash__(self):
        return hash('_'.join([self.name, self.label]))

    def __repr__(self):
        return '{} ({})'.format(self.name, self.label)

