"""
As described in:

    Petersen, H., Poon, J. Enhancing Short Text Clustering
    with Small External Repositories. 2011.

"""

import numpy as np
from scipy.spatial.distance import cdist
from broca.common.util import build_sim_mat
from broca.similarity.doc import DocSimilarity
from broca.knowledge.wikipedia import Wikipedia
from broca.tokenize.keywords import rake
from broca.vectorize.bow import BoW


class WikipediaSimilarity(Wikipedia, DocSimilarity):
    def __init__(self, tokenizer=rake, vectorizer=BoW, wiki_conn=None):
        super().__init__(wiki_conn=wiki_conn)
        self.vectorizer = vectorizer
        self.tokenizer = tokenizer

    def sim_mat(self, docs):
        tdocs = self.tokenizer(docs)
        all_terms = set([t for toks in tdocs for t in toks])
        print("fetching wikipedia pages...")
        bg_docs = [self.fetch_wikipage(t) for t in all_terms]

        print('vectorizing docs....')
        n_docs = len(docs)
        all_docs = docs + bg_docs

        print(all_docs)

        vectr = self.vectorizer()
        vecs = vectr.vectorize(all_docs).todense()

        doc_vecs = vecs[:n_docs] # target doc vecs
        bg_vecs = vecs[n_docs:] # background doc vecs

        # Bridging space representation of the docs
        doc_vecs = cdist(doc_vecs, bg_vecs, metric='cosine')
        print(doc_vecs)

        return build_sim_mat(doc_vecs, self.compute_bridge_similarity)

    def compute_bridge_similarity(self, vec1, vec2):
        EWP = 1 - np.multiply(vec1, vec2)

        # not sure exactly how to sort the EWP vector
        #EWP = sorted(EWP, reverse=True)
        EWP = sorted(EWP, reverse=True)

        k = 10
        EWP = EWP[:k]

        # The paper does not mention using logs but start to get into underflow
        # issues multiplying so many decimal values
        lEWP = -1 * np.log(EWP)
        print('lewp', lEWP)
        return 1/np.sum(lEWP)
