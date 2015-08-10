import numpy as np
from joblib import Parallel, delayed


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


def gram_size(term):
    """
    Convenience func for getting n-gram length.
    """
    return len(term.split(' '))


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


def sim_to_dist(sim_mat):
    """
    Convert a similarity matrix to a distance matrix.
    """
    return np.sqrt(1-sim_mat/np.max(sim_mat))


def dist_to_sim(dist_mat):
    """
    Approximately convert a distance matrix to a similarity matrix.
    """
    return 1-np.square(dist_mat/np.max(dist_mat))


def parallel(func, inputs, n_jobs, expand_args=False):
    """
    Convenience wrapper around joblib's parallelization.
    """
    if expand_args:
        return Parallel(n_jobs=n_jobs)(delayed(func)(*args) for args in inputs)
    else:
        return Parallel(n_jobs=n_jobs)(delayed(func)(arg) for arg in inputs)
