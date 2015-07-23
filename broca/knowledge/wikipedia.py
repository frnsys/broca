import re
from warnings import warn
from eatiht import v2 as eatiht
from urllib.error import HTTPError

wiki_footnote_re = re.compile(r'\[\s*[A-Za-z0-9\s]+\s*\]')
wikipedia = 'https://en.wikipedia.org/wiki/'


class Wikipedia():
    def __init__(self, wiki_conn=None):
        """
        Optionally pass a connection to a Wikipedia pages-article dump database.
        """
        self.wiki_conn = wiki_conn
        if self.wiki_conn is None:
            warn('You should use a Wikipedia pages-article dump instead of hitting the Wikipedia site')

    def fetch_wikipage(self, term):
        # TO DO use wiki_conn if available
        url = wikipedia + term.replace(' ', '_')

        try:
            text = eatiht.extract(url)

        # Page doesn't exist for this term
        except HTTPError:
            return ''

        text = wiki_footnote_re.sub(' ', text)
        return text
