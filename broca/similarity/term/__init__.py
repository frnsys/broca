"""
Term similarity classes provide similarity values for a pair of terms.
"""


class TermSimilarity():
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


from .wikipedia import WikipediaSimilarity
