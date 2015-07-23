from broca.common.util import gram_size


def prune(tdocs):
    """
    Prune terms which are totally subsumed by a phrase
    """
    all_terms = set([t for toks in tdocs for t in toks])
    redundant = {t for t in all_terms if gram_size(t) == 1}

    # This could be more efficient
    for doc in tdocs:
        cleared = set()
        for t in redundant:
            if t not in doc:
                continue

            # If this term occurs outside of a phrase,
            # it is no longer a candidate
            n = doc.count(t)
            d = sum(1 for t_ in doc if t != t_ and t in t_)
            if n > d:
                cleared.add(t)

        redundant = redundant.difference(cleared)

    pruned_tdocs = []
    for doc in tdocs:
        pruned_tdocs.append([t for t in doc if t not in redundant])

    return pruned_tdocs
