import re
from broca.common.util import parallel
from broca.preprocess import PreProcessor

try:
    from html.parser import HTMLParser
except ImportError: # Python 2
    from HTMLParser import HTMLParser

whitespace = re.compile(r'\s{2,}')


class HTMLCleaner(PreProcessor):
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def preprocess(self, docs):
        #print('HTML cleaning...')
        if self.n_jobs == 1:
            return [strip_html(d) for d in docs]
        else:
            return parallel(strip_html, docs, self.n_jobs)


def decode_html(s):
    """
    Returns the ASCII decoded version of the given HTML string. This does
    NOT remove normal HTML tags like <p>.
    from: <http://stackoverflow.com/a/275246/1097920>
    """
    for code in (
            ("'", '&#39;'),
            ('"', '&quot;'),
            ('>', '&gt;'),
            ('<', '&lt;'),
            ('&', '&amp;')
        ):
        s = s.replace(code[1], code[0])
    return s


def strip_html(html):
    # Any unwrapped text is ignored,
    # so wrap html tags just in case.
    # Looking for a more reliable way of stripping HTML...
    html = '<div>{0}</div>'.format(html)
    s = HTMLStripper()
    s.feed(html)

    # Stripping HTML adds additional whitespace, clean it up
    cleaned = s.get_data().strip()
    return whitespace.sub(' ', cleaned)


class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)
