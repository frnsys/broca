"""
Apriori Algorithm for text documents.

The original apriori algorithm was mainly for looking at "baskets" (i.e. shopping),
so some terminology may seem weird here. In particular, "transaction" refers to the set of tokens for a document.

See <https://en.wikipedia.org/wiki/Apriori_algorithm>
"""

from itertools import combinations
from collections import defaultdict
from broca.tokenize.keyword import POS
from broca.tokenize.util import prune
from broca.tokenize import Tokenizer


class Apriori(Tokenizer):
    def __init__(self, min_sup=0.5):
        self.min_sup = min_sup

    def tokenize(self, docs):
        """
        The first pass consists of converting documents
        into "transactions" (sets of their tokens)
        and the initial frequency/support filtering.

        Then iterate until we close in on a final set.

        `docs` can be any iterator or generator so long as it yields lists.
        Each list represents a document (i.e. is a list of tokens).
        For example, it can be a list of lists of nouns and noun phrases if trying
        to identify aspects, where each list represents a sentence or document.

        `min_sup` defines the minimum frequency (as a ratio over the total) necessary to
        keep a candidate.
        """
        if self.min_sup < 1/len(docs):
            raise Exception('`min_sup` must be greater than or equal to `1/len(docs)`.')

        # First pass
        candidates = set()
        transactions = []

        # Use nouns and noun phrases.
        for doc in POS().tokenize(docs):
            transaction = set(doc)
            candidates = candidates.union({(t,) for t in transaction})
            transactions.append(transaction)
        freq_set = filter_support(candidates, transactions, self.min_sup)

        # Iterate
        k = 2
        last_set = set()
        while freq_set != set():
            last_set = freq_set
            cands = generate_candidates(freq_set, k)
            freq_set = filter_support(cands, transactions, self.min_sup)
            k += 1

        # Map documents to their keywords.
        keywords = flatten(last_set)
        return prune([[kw for kw in keywords if kw in doc] for doc in docs])


def flatten(nested_tuple):
    """
    Flatten nested tuples.
    """
    return tuple([el for tupl in nested_tuple for el in tupl])


def filter_support(candidates, transactions, min_sup):
    """
    Filter candidates to a frequent set by some minimum support.
    """
    counts = defaultdict(lambda: 0)
    for transaction in transactions:
        for c in (c for c in candidates if set(c).issubset(transaction)):
            counts[c] += 1
    return {i for i in candidates if counts[i]/len(transactions) >= min_sup}


def generate_candidates(freq_set, k):
    """
    Generate candidates for an iteration.
    Use this only for k >= 2.
    """
    single_set = {(i,) for i in set(flatten(freq_set))}

    # TO DO generating all combinations gets very slow for large documents.
    # Is there a way of doing this without exhaustively searching all combinations?
    cands = [flatten(f) for f in combinations(single_set, k)]
    return [cand for cand in cands if validate_candidate(cand, freq_set, k)]


def validate_candidate(candidate, freq_set, k):
    """
    Checks if we should keep a candidate.
    We keep a candidate if all its k-1-sized subsets
    are present in the frequent sets.
    """
    for subcand in combinations(candidate, k-1):
        if subcand not in freq_set:
            return False
    return True
