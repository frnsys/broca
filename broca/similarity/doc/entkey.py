"""
A very experimental document similarity measurement
which looks at entity and keyword overlap.
"""

from broca import Pipe
from broca.common.util import build_sim_mat
from broca.similarity.doc import DocSimilarity


class EntKeySimilarity(DocSimilarity):
    input = Pipe.type.tokens, Pipe.type.entities

    def __init__(self, idf, idf_entity=None, term_sim_ref=None, debug=False):
        self.term_sim_ref = term_sim_ref
        self.idf = idf
        self.idf_entity = idf_entity if idf_entity is not None else idf
        self.debug = debug

    def __call__(self, token_docs, entities):
        pdocs = []
        for i, (tks, ents) in enumerate(zip(token_docs, entities)):
            pdocs.append(Document(i, ents, tks))

        return build_sim_mat(pdocs, self.similarity)

    def sim_mat(self, token_docs, entities):
        return self(token_docs, entities)

    def similarity(self, d, d_):
        """
        Compute a similarity score for two documents.

        Optionally pass in a `term_sim_ref` dict-like, which should be able
        to take `term1, term2` as args and return their similarity.
        """
        es = set([e.name for e in d.entities])
        es_ = set([e.name for e in d_.entities])
        e_weight = (len(es) + len(es_) - abs(len(es) - len(es_)))/2
        e_score = sum(self.idf_entity[t] for t in es & es_)

        toks = set(d.tokens)
        toks_ = set(d_.tokens)
        t_weight = (len(toks) + len(toks_) - abs(len(toks) - len(toks_)))/2

        # If no term similarity reference is passed,
        # look only at surface form overlap (i.e. exact overlap)
        shared_toks = toks & toks_
        overlap = [(t, t, self.idf[t]) for t in shared_toks]
        t_score = sum(self.idf[t] for t in shared_toks)
        if self.term_sim_ref is not None:
            # Double-count exact overlaps b/c we are
            # comparing bidirectional term pairs here
            t_score *= 2
            for toks1, toks2 in [(toks, toks_), (toks_, toks)]:
                for t in toks1 - shared_toks:
                    best_match = max(toks2, key=lambda t_: self.term_sim_ref[t, t_])
                    sim = self.term_sim_ref[t, best_match]
                    t_score += sim * ((self.idf[t] + self.idf[best_match])/2)
                    if sim > 0:
                        overlap.append((t, best_match, sim * ((self.idf[t] + self.idf[best_match])/2)))

            # Adjust term weight
            #t_weight /= 2

        t_weight = 1/t_weight if t_weight != 0 else 0
        e_weight = 1/e_weight if e_weight != 0 else 0
        t_score *= t_weight
        e_score *= e_weight

        if self.debug:
            print('\n-------------------------')
            print((d.id, d_.id))
            print('DOC:', d.id)
            print('DOC:', d_.id)
            print('\tEntities:')
            print('\t', es)
            print('\t', es_)
            print('\t\tEntity overlap:', es & es_)
            print('\t\tEntity weight:', e_weight)
            print('\t\tEntity score:', e_score)

            print('\tTokens:')
            print('\t\t', toks)
            print('\t\t', toks_)
            print('\t\tToken overlap:', overlap)
            print('\t\tToken weight:', t_weight)
            print('\t\tToken score:', t_score)

            print('\tTotal score:', t_score + e_score)

        return t_score + e_score



class Document():
    def __init__(self, id, entities, tokens):
        self.id = id
        self.entities = entities

        # Remove entity tokens from tokens so they aren't double-counted
        ent_names = [e.name for e in self.entities]
        self.tokens = [t for t in tokens if t not in ent_names]

    def __repr__(self):
        if hasattr(self, 'cluster'):
            return '{}_{}'.format(self.cluster, self.id)
        return '{}'.format(self.id)
