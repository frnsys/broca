import math
from collections import defaultdict


def compute_idf(t_docs, normalize=False):
    N = len(t_docs)
    iidf = defaultdict(int)
    for terms in t_docs:
        # Only care about presence, not frequency,
        # so convert to a set
        for t in set(terms):
            iidf[t] += 1

    for k, v in iidf.items():
        iidf[k] = math.log(N/v + 1, 10)

    if normalize:
        mxm = max(iidf.values())
        for k, v in iidf.items():
            iidf[k] = v/mxm

    return iidf
