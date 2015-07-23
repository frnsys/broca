from broca.pipeline import PipeType


class PreProcessor():
    input = PipeType.docs
    output = PipeType.docs

    def __call__(self, docs):
        return self.preprocess(docs)

    def preprocess(self, docs):
        raise NotImplementedError
