from scipy.spatial.distance import pdist, squareform
from broca.similarity.term import TermSimilarity
from broca.knowledge.wikipedia import Wikipedia
from broca.vectorize.bow import BoW


class WikipediaSimilarity(Wikipedia, TermSimilarity):
    def __init__(self, terms, wiki_conn=None):
        """
        Initialize with a list of terms.
        Will fetch Wikipedia pages for each term,
        if available, then compute their similarity matrix.
        """
        super().__init__(wiki_conn=wiki_conn)

        # Term map for similarity matrix lookup later
        terms = set(terms)
        self.term_map = {t: i for i, t in enumerate(terms)}

        # Fetch wikipages, compute cosine similarity matrix
        docs = [self.fetch_wikipage(t) for t in terms]
        vectr = BoW()
        vecs = vectr.vectorize(docs)
        dist_mat = pdist(vecs.todense(), metric='cosine')
        dist_mat = squareform(dist_mat)
        self.sim_mat = 1/(1 + dist_mat)

    def __getitem__(self, terms):
        t1, t2 = terms
        try:
            i1 = self.term_map[t1]
            i2 = self.term_map[t2]
            return self.sim_mat[i1, i2]

        # Term(s) not found
        except KeyError:
            return 0.
