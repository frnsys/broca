from broca.pipeline import Pipe


class Tokenizer(Pipe):
    input = Pipe.type.docs
    output = Pipe.type.tokens

    def __call__(self, docs):
        return self.tokenize(docs)

    def tokenize(self, docs):
        raise NotImplementedError


from .lemma import Lemma
