from gensim.models import Phrases
from sup.progress import Progress
from nltk.tokenize import sent_tokenize, word_tokenize


def train_phrases(paths, out='data/bigram_model.phrases', tokenizer=word_tokenize, **kwargs):
    """
    Train a bigram phrase model on a list of files.
    """
    n = 0
    for path in paths:
        print('Counting lines for {0}...'.format(path))
        n += sum(1 for line in open(path, 'r'))
    print('Processing {0} lines...'.format(n))

    # Change to use less memory. Default is 40m.
    kwargs = {
        'max_vocab_size': 40000000,
        'threshold': 8.
    }.update(kwargs)

    print('Training bigrams...')
    bigram = Phrases(_phrase_doc_stream(paths, n, tokenizer=word_tokenize), **kwargs)

    print('Saving...')
    bigram.save(out)


def _phrase_doc_stream(paths, n, tokenizer=word_tokenize):
    """
    Generator to feed sentences to the phrase model.
    """
    i = 0
    p = Progress()
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                i += 1
                p.print_progress(i/n)
                for sent in sent_tokenize(line.lower()):
                    tokens = tokenizer(sent)
                    yield tokens
