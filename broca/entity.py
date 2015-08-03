from broca import Pipe
from broca.common.shared import spacy


class Entities(Pipe):
    """
    Extracts entities and uses them as tokens.
    """
    input = Pipe.type.docs
    output = Pipe.type.entities

    def __call__(self, docs):
        entities = []
        for doc in docs:
            res = spacy(doc, entity=True, tag=True, parse=False)
            ents = [Entity(e.string, e.label_) for e in res.ents]
            entities.append(ents)
        return entities


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

