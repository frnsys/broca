import json
import math
from collections import defaultdict
from broca.common.util import parallel
from broca.knowledge.util import merge


def train_idf(tokens_stream, out=None, **kwargs):
    """
    Train a IDF model on a list of files (parallelized).
    """
    idfs = parallel(count_idf, tokens_stream, n_jobs=-1)
    N = len(idfs) # n docs
    idf = merge(idfs)

    for k, v in idf.items():
        idf[k] = math.log(N/v)
        # v ~= N/(math.e ** idf[k])

    # Keep track of N to update IDFs
    idf['_n_docs'] = N

    if out is not None:
        with open(out, 'w') as f:
            json.dump(idf, f)

    return idf


def count_idf(tokens):
    idf = defaultdict(int)
    # Don't count freq, just presence
    for token in set(tokens):
        idf[token] += 1
    return idf


class IDF():
    def __init__(self, path):
        self._idf = json.load(open(path, 'r'))

    def __getitem__(self, term):
        N = self._idf['_n_docs'] + 1
        return self._idf.get(term, math.log(N/1))

    def __contains__(self, term):
        return term in self._idf
