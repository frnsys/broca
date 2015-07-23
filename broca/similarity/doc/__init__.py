"""
Term similarity classes provide similarity values for a pair of documents.
"""


class DocSimilarity():
    """
    Doc similarity classes should inherit from this class
    """

    def __getitem__(self, docs):
        """
        Args:
            - docs -> a 2-tuple of strings

        Returns: a scalar value
        """
        raise NotImplementedError

    def sim_mat(self, docs):
        """
        Args:
            - docs -> a list of strings

        Returns: a similarity matrix
        """
        pass
    process = sim_mat


from .wikipedia import WikipediaSimilarity
