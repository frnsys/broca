"""
Vectorize classes provide vector representations
for a list of documents.
"""
from broca.pipeline import Pipe


class Vectorizer(Pipe):
    """
    Vectorizers should inherit from this class
    """
    input = Pipe.type.docs
    output = Pipe.type.vecs

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


from .bow import BoW
from .dcs import DCS
