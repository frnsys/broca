# -*- coding: utf-8 -*-

"""
Preprocess
----------

Each preprocessing method should take in a string
and return a string.
"""

from __future__ import unicode_literals

import re
import string


url_re = re.compile(r'https?:\/\/.*[\r\n]*', flags=re.MULTILINE)


def clean_doc(doc, remove_urls=True):
    doc = doc.lower()

    if remove_urls:
        # Remove URLs
        doc = url_re.sub('', doc)

    doc = doc.replace('\'s ', ' ')
    doc = strip_punct(doc)
    return doc.strip()


# Don't replace dashes, but do replace em-dashes with a space.
punct = (string.punctuation + '“”').replace('-', '')
dash_map = {ord(p): ' ' for p in '—'}
punct_map = {ord(p): '' for p in punct}
def strip_punct(doc):
    try:
        return doc.translate(dash_map).translate(punct_map)
    except TypeError: # Python 2
        trans = string.maketrans('', '')
        return doc.replace('–', ' ').translate(trans, punct)


