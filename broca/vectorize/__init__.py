"""
Vectorize classes provide vector representations
for a list of documents.
"""
from broca.pipeline import PipeType


class Vectorizer():
    """
    Vectorizers should inherit from this class
    """
    input = PipeType.docs
    output = PipeType.vecs

    def vectorize(self, docs):
        """
        Args:
            - docs -> a list or a generator of strings

        Returns: array of feature vectors
        """
        raise NotImplementedError

    def train(self, docs):
        """
        Args:
            - docs -> a list or a generator of strings
        """
        # Optional
        raise NotImplementedError

    def __call__(self, docs):
        return self.vectorize(docs)
