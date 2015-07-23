"""
Vectorize classes provide vector representations
for a list of documents.
"""

from broca.common.model import Model


class Vectorizer(Model):
    """
    Vectorizers should inherit from this class
    """

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
