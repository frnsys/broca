from broca.common.util import gram_size


def prune(tdocs):
    """
    Prune terms which are totally subsumed by a phrase

    This could be better if it just removes the individual keywords
    that occur in a phrase for each time that phrase occurs.
    """
    all_terms = set([t for toks in tdocs for t in toks])
    terms = set()
    phrases = set()
    for t in all_terms:
        if gram_size(t) > 1:
            phrases.add(t)
        else:
            terms.add(t)

    # Identify candidates for redundant terms (1-gram terms found in a phrase)
    redundant = set()
    for t in terms:
        if any(t in ph for ph in phrases):
            redundant.add(t)

    # Search all documents to check that these terms occur
    # only in a phrase. If not, remove it as a candidate.
    # This could be more efficient
    cleared = set()
    for t in redundant:
        if any(check_term(d, term=t) for d in tdocs):
            cleared.add(t)

    redundant = redundant.difference(cleared)

    pruned_tdocs = []
    for doc in tdocs:
        pruned_tdocs.append([t for t in doc if t not in redundant])

    return pruned_tdocs


def check_term(tdoc, term):
    if term not in tdoc:
        return False

    # If this term occurs outside of a phrase,
    # it is no longer a candidate
    n = tdoc.count(term)
    # Count phrases that contain t
    d = sum(1 for ph in tdoc if term != ph and term in ph)
    return n > d

def check_phrase(phrase, term):
    return term in phrase
