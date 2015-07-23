from broca.pipeline import PipeType


class Tokenizer():
    input = PipeType.docs
    output = PipeType.tokens

    def __call__(self, docs):
        return self.tokenize(docs)

    def tokenize(self, docs):
        raise NotImplementedError


from .lemma import Lemma
