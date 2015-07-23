"""
A naive keyword extractor which just pulls out nouns and noun phrases.

Was using the PerceptronTagger is _way_ faster than NLTK's default tagger, and more accurate to boot.
See <http://stevenloria.com/tutorial-state-of-the-art-part-of-speech-tagging-in-textblob/>.

However, it complicates the library's installation, and the spacy tagger is quite fast and good too.
"""

from broca.common import spacy

CFG = {
    ('NNP', 'NNP'): 'NNP',
    ('NN', 'NN'): 'NNI',
    ('NNI', 'NN'): 'NNI',
    ('JJ', 'JJ'): 'JJ',
    ('JJ', 'NN'): 'NNI',
}


def extract_keywords(docs):
    tags = ['NN', 'NNS', 'NNP', 'NNPS']

    keywords = []
    for doc in docs:
        toks = spacy(doc, tag=True, parse=False, entity=False)
        tagged = [(t.lower_.strip(), t.tag_) for t in toks]
        kws = [t for t, tag in tagged if tag in tags]
        kws += extract_noun_phrases(tagged)
        keywords.append(kws)
    return keywords


def extract_noun_phrases(tagged_doc):
    """
    (From textblob)
    """
    tags = _normalize_tags(tagged_doc)
    merge = True
    while merge:
        merge = False
        for x in range(0, len(tags) - 1):
            t1 = tags[x]
            t2 = tags[x + 1]
            key = t1[1], t2[1]
            value = CFG.get(key, '')
            if value:
                merge = True
                tags.pop(x)
                tags.pop(x)
                match = '%s %s' % (t1[0], t2[0])
                pos = value
                tags.insert(x, (match, pos))
                break

    matches = [t[0] for t in tags if t[1] in ['NNP', 'NNI']]
    return matches


def _normalize_tags(chunk):
    """
    (From textblob)

    Normalize the corpus tags.
    ("NN", "NN-PL", "NNS") -> "NN"
    """
    ret = []
    for word, tag in chunk:
        if tag == 'NP-TL' or tag == 'NP':
            ret.append((word, 'NNP'))
            continue
        if tag.endswith('-TL'):
            ret.append((word, tag[:-3]))
            continue
        if tag.endswith('S'):
            ret.append((word, tag[:-1]))
            continue
        ret.append((word, tag))
    return ret
