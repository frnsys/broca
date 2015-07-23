import string

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from broca.tokenize import Tokenizer


class Lemma(Tokenizer):
    """
    Lemmatizing tokenizer.
    """
    def __init__(self):
        self.lemmr = WordNetLemmatizer()
        self.stops = stopwords.words('english')
        self.punct = {ord(p): ' ' for p in string.punctuation + '“”'}

        # Treat periods specially, replacing them with nothing.
        # This is so that initialisms like F.D.A. get preserved as FDA.
        self.period = {ord('.'): None}

    def tokenize(self, docs):
        """ Tokenizes a document, using a lemmatizer.

        Args:
            | doc (str)                 -- the text document to process.

        Returns:
            | list                      -- the list of tokens.
        """
        tokens = []

        for doc in docs:
            toks = []

            # Strip punctuation.
            doc = doc.translate(self.period)
            doc = doc.translate(self.punct)

            for token in word_tokenize(doc):
                # Ignore punctuation and stopwords
                if token in self.stops:
                    continue

                # Lemmatize
                lemma = self.lemmr.lemmatize(token.lower())
                toks.append(lemma)

            tokens.append(toks)

        return tokens
