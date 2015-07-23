"""
Wikipedia-based term similarity
"""
from broca.similarity.term import TermSimilarity


class WikipediaSimilarity(TermSimilarity):
    def __init__(self, terms):
        """
        Initialize with a list of terms.
        Will fetch Wikipedia pages for each term,
        if available, then compute their similarity matrix.
        """
        pass

    def __getitem__(self, terms):
        pass
