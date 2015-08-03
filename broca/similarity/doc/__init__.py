"""
Term similarity classes provide similarity values for a pair of documents.
"""

from broca import Pipe


class DocSimilarity(Pipe):
    """
    Doc similarity classes should inherit from this class
    """
    output = Pipe.type.sim_mat

    def sim_mat(self, docs):
        """
        Args:
            - docs -> a list of strings

        Returns: a similarity matrix
        """
        raise NotImplementedError


from .wikipedia import WikipediaSimilarity
from .entkey import EntKeySimilarity
