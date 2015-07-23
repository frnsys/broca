import math
import numpy as np
from collections import defaultdict


def penn_to_wordnet(tag):
    """
    Convert a Penn Treebank PoS tag to WordNet PoS tag.
    """
    if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
        return 'n' #wordnet.NOUN
    elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return 'v' #wordnet.VERB
    elif tag in ['RB', 'RBR', 'RBS']:
        return 'r' #wordnet.ADV
    elif tag in ['JJ', 'JJR', 'JJS']:
        return 'a' #wordnet.ADJ
    return None


def idf(t_docs):
    N = len(t_docs)
    iidf = defaultdict(int)
    for terms in t_docs:
        # Only care about presence, not frequency,
        # so convert to a set
        for t in set(terms):
            iidf[t] += 1

    for k, v in iidf.items():
        iidf[k] = math.log(N/v + 1, 10)

    # Normalize
    mxm = max(iidf.values())
    for k, v in iidf.items():
        iidf[k] = v/mxm

    return iidf


def build_sim_mat(items, sim_func):
    n = len(items)
    sim_mat = np.zeros((n, n))

    for i, d1 in enumerate(items):
        for j, d2 in enumerate(items):
            if i == j:
                sim_mat[i,j] = 1.

            # Just build the lower triangle
            # (assuming symmetric similarity)
            elif i > j:
                sim_mat[i,j] = sim_func(d1, d2)

    # Construct the full sim mat from the lower triangle
    return sim_mat + sim_mat.T - np.diag(sim_mat.diagonal())
