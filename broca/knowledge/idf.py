import json
import math
from functools import partial
from collections import defaultdict
from sup.parallel import parallelize
from nltk.tokenize import word_tokenize
from broca.knowledge.util import merge, split_file, doc_stream


def train_idf(paths, out='data/idf.json', tokenizer=word_tokenize, **kwargs):
    """
    Train a IDF model on a list of files (parallelized).
    """
    print('Preparing files...')
    args = []
    for path in paths:
        args += [(file,) for file in split_file(path, chunk_size=5000)]

    print('Counting terms...')
    p_count_idf = partial(count_idf, tokenizer=tokenizer)
    results = parallelize(p_count_idf, args)

    idfs, n_docs = zip(*results)

    print('Merging...')
    idf = merge(idfs)

    print('Computing IDFs...')
    N = sum(n_docs)
    for k, v in idf.items():
        idf[k] = math.log(N/v)
        # v ~= N/(math.e ** idf[k])

    # Keep track of N to update IDFs
    idf['_n_docs'] = N

    with open(out, 'w') as f:
        json.dump(idf, f)


def count_idf(path, tokenizer):
    N = 0
    idf = defaultdict(int)
    for tokens in doc_stream(path):
        N += 1
        # Don't count freq, just presence
        for token in set(tokens):
            idf[token] += 1
    return idf, N


class IDF():
    def __init__(self, path):
        self._idf = json.load(open(path, 'r'))

    def __getitem__(self, term):
        N = self._idf['_n_docs'] + 1
        return self._idf.get(term, math.log(N/1))

    def __contains__(self, term):
        return term in self._idf
