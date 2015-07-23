# -*- coding: utf-8 -*-

try:
    intern('')
except NameError:
    from sys import intern
import string
import random
from nltk.tokenize import sent_tokenize
from broca.generate import Generator


class Markov(Generator):
    def __init__(self, ngram_size=1, max_chars=None, ramble=True, spasm=0.05, filepath='markov.pickle'):
        """
        ngram_size (int)
        Size of ngrams to use for knowledge. on smaller datasets, a value of 1 is recommended,
        you will get more incoherent blabber, but things won't stop short.
        If you have a large dataset, use a value like 2 or 3. More data at this ngram size means higher quality.

        max_chars (int)
        Maximum characters the generated speech should be under.

        ramble (bool)
        Whether or not to randomly pick a next token if the generator gets stuck.

        spasm (float)
        The probability of "spasming", that is, picking a random token as the next token.
        This can make things more interesting!

        filepath (str)
        Where to save/load the Markov to/from.
        """
        self.n = ngram_size
        self.max_chars = max_chars
        self.ramble = ramble
        self.filepath = filepath
        self.spasm = spasm
        self.stop_token = '<STOP>'

        # For limiting recursion-generating of new speech.
        self.max_retries = 100
        self.retries = 0

        # In the format of:
        # ('this', 'is', 'an'): [1, 'example']
        self.knowledge = {}

        # For keeping track of good starting tokens.
        # Starting tokens come after a prev value of (),
        # that is, after no previous tokens.
        self.knowledge[()] = {}

        # Keep track of the last ngram seen,
        # so we can pick the next token.
        self.prev = ()


    def train(self, docs, stop_rule=lambda token: False):
        """
        Add to knowledge the learnings
        from some input docs.

        docs (list)
        The documents to train on.

        stop_rule (callable: (x) => bool)
        A callable which takes a token as input and returns True if it should be skipped.
        """
        for doc in docs:
            for sent in sent_tokenize(doc):
                tokens = self._tokenize(sent, stop_rule)
                if tokens:
                    # Keep track of starting token candidates.
                    start_token = tuple(tokens[0:self.n])
                    self.knowledge[()][start_token] = self.knowledge[()].get(start_token, 0) + 1

                    # Example, where ngram_size=3:
                    # ngram = ['this', 'is', 'an', 'example']
                    for ngram in self._ngramize(tokens):

                        # The tokens leading up to the 'post'.
                        # e.g. ('this', 'is', 'an')
                        prior = tuple(ngram[:self.n])

                        # The 'post' token.
                        # e.g. 'example'
                        post = ngram[-1]

                        # Get existing data, if it's there.
                        # Keeps track as:
                        # prior: {post: count}

                        # Create new prior entry if necessary.
                        if prior not in self.knowledge:
                            self.knowledge[prior] = {}

                        # Create new token entry for this prior
                        # if necessary.
                        if post not in self.knowledge[prior]:
                            self.knowledge[prior][post] = 0

                        # Increment count of this post token
                        # for this prior
                        self.knowledge[prior][post] += 1


    def speak(self):
        """
        Generate some 'speech'.
        """

        # Reset the previous tokens.
        self.prev = ()

        tokens = []

        def constraint(tokens):
            if self.max_chars is None:
                return True
            return len(' '.join(tokens)) < self.max_chars

        while constraint(tokens):
            next_token = self._next_token()

            # If a token couldn't be found, or
            # if the next token is a stop token,
            # stop.
            if not next_token or next_token == self.stop_token:
                break

            # Update the prev tokens,
            # truncating if necessary.
            if type(next_token) is tuple:
                self.prev += next_token
                tokens += list(next_token)
            else:
                self.prev += (next_token,)
                tokens.append(next_token)


            if len(self.prev) > self.n:
                self.prev = self.prev[1:]

        # If the constraint is violated, try consolidating.
        if not constraint(tokens):
            full = tokens
            consolidated = self._consolidate(tokens)

            if not consolidated:
                # Try to generate a new string.
                if self.retries < self.max_retries:
                    self.retries += 1
                    tokens = self.generate().split(' ')
                else:
                    # If this max retries has been hit,
                    # just drop the last token.
                    tokens = full[:-1]
            else:
                tokens = consolidated

        self.retries = 0
        return ' '.join(tokens)


    def _ngramize(self, tokens):
        """
        A generator which chunks a list of tokens
        into ngram lists.
        """

        # Ensure we have enough tokens to work with.
        if len(tokens) > self.n:
            # Add the <stop> token to the end.
            tokens.append(self.stop_token)

            # Yield the ngrams.
            for i in range(len(tokens) - self.n):
                next = i + self.n + 1
                yield tokens[i:next]


    def _tokenize(self, doc, stop_rule):
        """
        Tokenizes a document.
        This is a very naive tokenizer;
        i.e. it has no stop words,
        since we need those words to generate convincing speech.
        It also strips punctuation from the beginning and end of tokens,
        except for '@' at the beginning of a token.

        Optionally provide a `stop_rule` function,
        which should return True if a token should be stopped on.
        """
        tokens = []
        punctuation = string.punctuation.replace('@', '') + '“”‘’–"'

        for token in doc.split(' '):
            # This saves memory by having
            # duplicate strings just point to the same memory.
            token = intern(token.strip(punctuation))

            # Ignore punctuation and stopwords
            if not token or stop_rule(token):
                continue

            tokens.append(token.lower())
        return tokens


    def _consolidate(self, tokens):
        """
        Shortens the generated text so it is
        less than the max character length.

        The strategy is to remove ngrams until
        the list of tokens ends with an ngram that
        has a <STOP> associated with it.
        The truncated list of tokens is returned.
        This list could potentially be empty.
        """
        # The n-length tail of the generated text,
        # to see if it's a stop candidate.
        tail = tuple(tokens[-self.n:])

        # Check if we can stop here.
        # Stop here anyway if tail is (), since it will infinitely recurse,
        # and it implies the tokens list is now empty.
        if not tail or (self.knowledge.get(tail, {}).get('<STOP>', 0) and len(tokens) < self.max_chars):
            return tokens

        # If not, RECURSE!
        else:
            tokens = tokens[0:-self.n]
            return self._consolidate(tokens)


    def _next_token(self):
        """
        Choose the next token.

        If there's a key error, it may be
        because self.prev doesn't have enough grams/tokens in it,
        which  means we're still early in the generation, so just pick
        a random starting token.
        Otherwise, it's probably because the self.prev ngram has never
        been enconutered before. In which case, if self.ramble is True,
        pick a random starting token, otherwise, just end return None.
        """
        if random.random() < self.spasm:
            return self._weighted_choice(self.knowledge[()])
        else:
            try:
                return self._weighted_choice(self.knowledge[self.prev])
            except KeyError:
                if len(self.prev) < self.n or self.ramble:
                    return self._weighted_choice(self.knowledge[()])


    def _weighted_choice(self, choices):
        """
        Random selects a key from a dictionary,
        where each key's value is its probability weight.
        """
        # Randomly select a value between 0 and
        # the sum of all the weights.
        rand = random.uniform(0, sum(choices.values()))

        # Seek through the dict until a key is found
        # resulting in the random value.
        summ = 0.0
        for key, value in choices.items():
            summ += value
            if rand < summ: return key

        # If this returns False,
        # it's likely because the knowledge is empty.
        return False