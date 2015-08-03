from broca import Pipe


class PreProcessor(Pipe):
    input = Pipe.type.docs
    output = Pipe.type.docs

    def __call__(self, docs):
        return self.preprocess(docs)

    def preprocess(self, docs):
        raise NotImplementedError


from .clean import BasicCleaner
from .html import HTMLCleaner
