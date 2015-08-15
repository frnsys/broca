"""
For manipulating text.
"""

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from broca.vectorize import Vectorizer
from broca.tokenize import LemmaTokenizer


class Tokenizer():
    """
    Wrap broca tokenizers
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, doc):
        return self.tokenizer.tokenize([doc])[0]


class BoWVectorizer(Vectorizer):
    def __init__(self, min_df=1, max_df=0.9, tokenizer=LemmaTokenizer, hash=False):
        """
        `min_df` is set to filter out extremely rare words,
        since we don't want those to dominate the distance metric.

        `max_df` is set to filter out extremely common words,
        since they don't convey much information.
        """

        # Wrap the specified tokenizer
        t = Tokenizer(tokenizer())

        if hash:
            vectr = HashingVectorizer(input='content', stop_words='english', lowercase=True, tokenizer=t)
        else:
            vectr = CountVectorizer(input='content', stop_words='english', lowercase=True, tokenizer=t, min_df=min_df, max_df=max_df)

        args = [
            ('vectorizer', vectr),
            ('tfidf', TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)),
            ('normalizer', Normalizer(copy=False))
        ]

        self.pipeline = Pipeline(args)
        self.trained = False

    def vectorize(self, docs):
        if not self.trained:
            return self.train(docs)
        return self.pipeline.transform(docs)

    def train(self, docs):
        self.trained = True
        return self.pipeline.fit_transform(docs)

    @property
    def vocabulary(self):
        return self.pipeline.named_steps['vectorizer'].get_feature_names()
