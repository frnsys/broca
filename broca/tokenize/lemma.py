from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from broca.tokenize import Tokenizer
from broca.common.shared import spacy
from broca.common.util import penn_to_wordnet, parallel


class LemmaTokenizer(Tokenizer):
    """
    Lemmatizing tokenizer.
    """
    def __init__(self, n_jobs=1):
        self.lemmr = WordNetLemmatizer()
        self.stops = stopwords.words('english')
        self.n_jobs = n_jobs

    def tokenize(self, docs):
        """ Tokenizes a document, using a lemmatizer.

        Args:
            | doc (str)                 -- the text document to process.

        Returns:
            | list                      -- the list of tokens.
        """
        if self.n_jobs == 1:
            return [self._tokenize(doc) for doc in docs]
        else:
            return parallel(self._tokenize, docs, self.n_jobs)

    def _tokenize(self, doc):
        toks = []

        for t in spacy(doc, tag=True, parse=False, entity=False):
            token = t.lower_.strip()
            tag = t.tag_

            # Ignore stopwords
            if token in self.stops:
                continue

            # Lemmatize
            wn_tag = penn_to_wordnet(tag)
            if wn_tag is not None:
                lemma = self.lemmr.lemmatize(token, wn_tag)
                toks.append(lemma)
            else:
                toks.append(token)
        return toks
