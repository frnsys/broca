from functools import partial
from broca.tokenize import Tokenizer
from broca.common.shared import spacy
from broca.common.util import parallel, penn_to_wordnet
from broca.tokenize.keyword.rake import RAKETokenizer
from gensim.models import Phrases
from nltk.stem.wordnet import WordNetLemmatizer


class OverkillTokenizer(Tokenizer):
    def __init__(self, lemmatize=True, n_jobs=1, bigram=None, trigram=None, min_count=5, threshold=10.):
        self.lemmatize = lemmatize
        self.n_jobs = n_jobs
        self.bigram = bigram
        self.trigram = trigram
        self.min_count = min_count
        self.threshold = threshold

    def tokenize(self, docs):
        if self.lemmatize:
            lem = WordNetLemmatizer()

        #print('RAKE tokenizing...')
        pre_tdocs = RAKETokenizer(n_jobs=self.n_jobs).tokenize(docs)

        for i, tdoc in enumerate(pre_tdocs):
            for t in tdoc:
                if t.startswith('one'):
                    print(t)
                    print(i)

        #print('Additional Tokenizing docs...')
        if self.n_jobs == 1:
            tdocs = [pre_tokenize(doc, tdoc, lem=lem) for doc, tdoc in zip(docs, pre_tdocs)]
        else:
            tdocs = parallel(partial(pre_tokenize, lem=lem), zip(docs, pre_tdocs), self.n_jobs, expand_args=True)

        #print('Training bigram...')
        if self.bigram is None:
            self.bigram = Phrases(tdocs,
                                  min_count=self.min_count,
                                  threshold=self.threshold,
                                  delimiter=b' ')
        else:
            self.bigram.add_vocab(tdocs)

        #print('Training trigram...')
        if self.trigram is None:
            self.trigram = Phrases(self.bigram[tdocs],
                                   min_count=self.min_count,
                                   threshold=self.threshold,
                                   delimiter=b' ')
        else:
            self.trigram.add_vocab(self.bigram[tdocs])

        return [tdoc for tdoc in self.trigram[self.bigram[tdocs]]]


def pre_tokenize(doc, tdoc, lem):
    # Split phrase keywords into 1gram keywords,
    # to check tokens against
    # We learn keyphrases later on.
    kws_1g = [t.split(' ') for t in tdoc]
    kws_1g = [kw for grp in kws_1g for kw in grp]

    toks = spacy(doc, tag=True, parse=False, entity=False)
    tagged = [(t.lower_.strip(), t.tag_) for t in toks]

    toks = []
    for tok, tag in tagged:
        if tok in kws_1g:
            wn_tag = penn_to_wordnet(tag)
            if wn_tag is not None:
                toks.append(lem.lemmatize(tok, wn_tag))

    return toks
