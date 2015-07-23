"""
Term similarity classes provide similarity values for a pair of documents.
"""
from broca.common.model import Model


class DocSimilarity(Model):
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
