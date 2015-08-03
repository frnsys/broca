from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from broca.tokenize import Tokenizer
from broca.common.shared import spacy
from broca.common.util import penn_to_wordnet


class LemmaTokenizer(Tokenizer):
    """
    Lemmatizing tokenizer.
    """
    def __init__(self):
        self.lemmr = WordNetLemmatizer()
        self.stops = stopwords.words('english')

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

            tokens.append(toks)

        return tokens
