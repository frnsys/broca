"""
Term similarity classes provide similarity values for a pair of terms.
"""
from broca.common.model import Model


class TermSimilarity(Model):
    """
    Term similarity classes should inherit from this class
    """

    def __getitem__(self, terms):
        """
        Args:
            - terms -> a 2-tuple of strings

        Returns: a scalar value
        """
        raise NotImplementedError


from broca.similarity.term.wikipedia import WikipediaSimilarity
