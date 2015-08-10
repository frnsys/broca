# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import string
from functools import partial
from broca.common.util import parallel
from broca.preprocess import PreProcessor

# Don't replace dashes.
# Everything else is replaced with a space in case no space is around the
# punctuation (which would cause unwanted term merging)
punct = (string.punctuation + '“”–').replace('-', '')
strip_map = {ord(p): '' for p in '\''}
punct_map = {ord(p): ' ' for p in punct}
url_re = re.compile(r'https?:\/\/.*[\r\n]*', flags=re.MULTILINE)
whs_re = re.compile(r'\W{2,}')


class BasicCleaner(PreProcessor):
    def __init__(self, remove_urls=True, lowercase=True, remove_possessors=True, remove_punctuation=True, n_jobs=1):
        self.remove_urls = remove_urls
        self.lowercase = lowercase
        self.remove_possessors = remove_possessors
        self.remove_punctuation = remove_punctuation
        self.n_jobs = n_jobs

    def preprocess(self, docs):
        print('Cleaning...')
        clean_func = partial(
            clean,
            remove_urls=self.remove_urls,
            lowercase=self.lowercase,
            remove_possessors=self.remove_possessors,
            remove_punctuation=self.remove_punctuation
        )
        if self.n_jobs == 1:
            return [clean_func(d) for d in docs]
        else:
            return parallel(clean_func, docs, self.n_jobs)


def clean(doc, remove_urls=True, lowercase=True, remove_possessors=True, remove_punctuation=True):
    if lowercase:
        doc = doc.lower()

    if remove_urls:
        # Remove URLs
        doc = url_re.sub('', doc)

    if remove_possessors:
        doc = doc.replace('\'s ', ' ')

    if remove_punctuation:
        doc = strip_punct(doc)

    return doc.strip()


def strip_punct(doc):
    doc = doc.translate(strip_map).translate(punct_map)

    # Collapse whitespace to single whitespace
    return whs_re.sub(' ', doc)
