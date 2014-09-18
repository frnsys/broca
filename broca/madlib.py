import re
import random
import pickle
import datetime

from nltk import word_tokenize, pos_tag

from .model import Model

ELIGIBLE_TAGS = [
    'CD',   # numbers
    'JJ',   # adjectives
    'NN',   # nouns
    'NNP',  # proper nouns
    'NNPS', # plural proper nouns
    'NNS',  # plural nouns
    'VBN',
    'VBG',
    'VB',
    'RB'    # adverbs
]

class Madlib(Model):
    def __init__(self, eligible_tags=ELIGIBLE_TAGS, filepath='madlib.pickle'):
        # Mad-lib patterns.
        self.patterns = []

        # { POS tag: words }
        self.vocabulary = {}

        # Tags which will be replaced.
        self.eligible_tags = eligible_tags


    def train(self, docs):
        for doc in docs:
            pattern = doc

            # Extract parts of speech.
            tokens = word_tokenize(doc)
            for t in pos_tag(tokens):
                token = t[0]
                tag = t[1]

                # Preserve hashtags.
                # Skip untagged tokens.
                # Skip tokens which are too short.
                if tag == '-NONE-' or len(token) <= 2:
                    continue

                if tag in self.eligible_tags:
                    # Build the pattern.
                    pattern = pattern.replace(token, '{{{{ {0} }}}}'.format(tag))

                    # Add new tokens to the vocabulary.
                    if tag not in self.vocabulary:
                        self.vocabulary[tag] = []
                    self.vocabulary[tag].append(token.lower())

            self.patterns.append(pattern)


    def speak(self):
        pattern = random.choice(self.patterns)
        output = pattern

        # Extract the tags to be replaced.
        p = re.compile(r'\{\{\s*([A-Za-z]+)\s*\}\}')
        tags = p.findall(pattern)

        # Replace the tags with selections from the vocabulary.
        for tag in tags:
            token = random.choice(self.vocabulary[tag])
            output = re.sub(r'(\{\{\s*' + re.escape(tag) + r'\s*\}\})', token, output, 1)
        return output
