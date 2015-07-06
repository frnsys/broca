import math
from collections import defaultdict
from functools import wraps


def handle_args(f):
    """
    Converts a single-string argument to a list-of-strings argument
    and handles the output accordingly.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if not isinstance(args[0], list):
            args = list(args)
            args[0] = [args[0]]
            return f(*args, **kwargs)[0]
        else:
            return f(*args, **kwargs)
    return decorated


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
