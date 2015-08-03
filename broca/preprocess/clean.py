# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import string
from broca.preprocess import PreProcessor

# Don't replace dashes, but do replace em-dashes with a space.
punct = (string.punctuation + '“”').replace('-', '')
dash_map = {ord(p): ' ' for p in '—'}
punct_map = {ord(p): '' for p in punct}
url_re = re.compile(r'https?:\/\/.*[\r\n]*', flags=re.MULTILINE)


class BasicCleaner(PreProcessor):
    def __init__(self, remove_urls=True, lowercase=True, remove_possessors=True, remove_punctuation=True):
        self.remove_urls = remove_urls
        self.lowercase = lowercase
        self.remove_posessors = remove_possessors
        self.remove_punctuation = remove_punctuation

    def preprocess(self, docs):
        return [clean(d,
                      self.remove_urls,
                      self.lowercase,
                      self.remove_posessors,
                      self.remove_punctuation) for d in docs]


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
    try:
        return doc.translate(dash_map).translate(punct_map)
    except TypeError: # Python 2
        trans = string.maketrans('', '')
        return doc.replace('–', ' ').translate(trans, punct)


