import re
from collections import defaultdict
from nltk.stem.wordnet import WordNetLemmatizer
from broca.tokenize import Tokenizer
from broca.tokenize.util import prune
from broca.common.util import gram_size, penn_to_wordnet
from broca.tokenize.keyword.rake import RAKETokenizer
from broca.common.shared import spacy


class OverkillTokenizer(Tokenizer):
    def __init__(self, prune=True, lemmatize=True):
        self.prune = prune
        self.lemmatize = lemmatize

    def tokenize(self, docs):
        if self.lemmatize:
            lem = WordNetLemmatizer()

        pre_tdocs = RAKETokenizer().tokenize(docs)

        tdocs = []
        for i, tdoc in enumerate(pre_tdocs):
            # Split phrase keywords into 1gram keywords,
            # to check tokens against
            kws_1g = [t.split(' ') for t in tdoc]
            kws_1g = [kw for grp in kws_1g for kw in grp]

            toks = spacy(docs[i], tag=True, parse=False, entity=False)
            tagged = [(t.lower_.strip(), t.tag_) for t in toks]

            toks = []
            for tok, tag in tagged:
                if tok in kws_1g:
                    wn_tag = penn_to_wordnet(tag)
                    if wn_tag is not None:
                        toks.append(lem.lemmatize(tok, wn_tag))
            tdocs.append(toks)

        tdocs = extract_phrases(tdocs, docs)
        if prune:
            return prune(tdocs)
        return tdocs


def extract_phrases(tdocs, docs):
    """
    Learn novel phrases by looking at co-occurrence of candidate term pairings.
    Docs should be input in tokenized (`tdocs`) and untokenized (`docs`) form.
    """
    # Gather existing keyphrases
    keyphrases = set()
    for doc in tdocs:
        for t in doc:
            if gram_size(t) > 1:
                keyphrases.add(t)

    # Count document co-occurrences
    t_counts = defaultdict(int)
    pair_docs = defaultdict(list)
    for i, terms in enumerate(tdocs):
        # We dont convert the doc to a set b/c we want to preserve order
        # Iterate over terms as pairs
        for pair in zip(terms, terms[1:]):
            t_counts[pair] += 1
            pair_docs[pair].append(i)

    # There are a lot of co-occurrences, filter down to those which could
    # potentially be phrases.
    t_counts = {kw: count for kw, count in t_counts.items() if count >= 2}

    # Identify novel phrases by looking at
    # keywords which co-occur some percentage of the time.
    # This could probably be more efficient/cleaned up
    for (kw, kw_), count in t_counts.items():
        # Look for phrases that are space-delimited or joined by 'and' or '-'
        ph_reg = re.compile('({0}|{1})(\s|-)(and\s)?({0}|{1})'.format(kw, kw_))

        # Extract candidate phrases and keep track of their counts
        phrases = defaultdict(int)
        phrase_docs = defaultdict(set)
        for i in pair_docs[(kw, kw_)]:
            for m in ph_reg.findall(docs[i].lower()):
                phrases[''.join(m)] += 1
                phrase_docs[''.join(m)].add(i)

        if not phrases:
            continue

        # Get the phrase encountered the most
        top_phrase = max(phrases.keys(), key=lambda k: phrases[k])
        top_count = phrases[top_phrase]

        if top_count/count >= 0.8:
            # Check if this new phrase is contained by an existing keyphrase.
            if any(top_phrase in ph for ph in keyphrases):
                continue
            keyphrases.add(top_phrase)

            # Add the new phrase to each doc it's found in
            for i in phrase_docs[top_phrase]:
                tdocs[i].append(top_phrase)

    return tdocs

