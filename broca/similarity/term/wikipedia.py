"""
Wikipedia-based term similarity
"""
import re
from warnings import warn
from eatiht import v2 as eatiht
from urllib.error import HTTPError
from scipy.spatial.distance import pdist, squareform
from broca.similarity.term import TermSimilarity
from broca.vectorize.bow import BoW

wiki_footnote_re = re.compile(r'\[\s*[A-Za-z0-9\s]+\s*\]')
wikipedia = 'https://en.wikipedia.org/wiki/'


class WikipediaSimilarity(TermSimilarity):
    def __init__(self, terms, wiki_conn=None):
        """
        Initialize with a list of terms.
        Will fetch Wikipedia pages for each term,
        if available, then compute their similarity matrix.

        Optionally pass a connection to a Wikipedia pages-article dump database.
        """
        self.wiki_conn = wiki_conn
        if self.wiki_conn is None:
            warn('You should use a Wikipedia pages-article dump instead of hitting the Wikipedia site')

        # Term map for similarity matrix lookup later
        terms = set(terms)
        self.term_map = {t: i for i, t in enumerate(terms)}

        # Fetch wikipages, compute cosine similarity matrix
        docs = [self._fetch_wikipage(t) for t in terms]
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

    def _fetch_wikipage(self, term):
        # TO DO use wiki_conn if available
        url = wikipedia + term.replace(' ', '_')

        try:
            text = eatiht.extract(url)

        # Page doesn't exist for this term
        except HTTPError:
            return ''

        text = wiki_footnote_re.sub(' ', text)
        return text
