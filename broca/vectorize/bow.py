"""
For manipulating text.
"""

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from broca.vectorize import Vectorizer
from broca.tokenize import Lemma


class BoW(Vectorizer):
    def __init__(self, min_df=0.015, max_df=0.9, tokenizer=Lemma):
        """
        `min_df` is set to filter out extremely rare words,
        since we don't want those to dominate the distance metric.

        `max_df` is set to filter out extremely common words,
        since they don't convey much information.
        """

        # Wrap the specified tokenizer
        t = tokenizer()
        class Tokenizer():
            def __call__(self, doc):
                return t.tokenize([doc])[0]

        args = [
            ('vectorizer', CountVectorizer(input='content', stop_words='english', lowercase=True, tokenizer=Tokenizer(), min_df=min_df, max_df=max_df)),
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
