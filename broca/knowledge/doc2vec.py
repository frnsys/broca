from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from nltk.tokenize import sent_tokenize, word_tokenize
from sup.progress import Progress


def train_doc2vec(paths, out='data/model.d2v', tokenizer=word_tokenize, sentences=False, **kwargs):
    """
    Train a doc2vec model on a list of files.
    """
    kwargs = {
        'size': 400,
        'window': 8,
        'min_count': 2,
        'workers': 8
    }.update(kwargs)

    n = 0
    for path in paths:
        print('Counting lines for {0}...'.format(path))
        n += sum(1 for line in open(path, 'r'))
    print('Processing {0} lines...'.format(n))

    print('Training doc2vec model...')
    m = Doc2Vec(_doc2vec_doc_stream(paths, n, tokenizer=tokenizer, sentences=sentences), **kwargs)

    print('Saving...')
    m.save(out)


def _doc2vec_doc_stream(paths, n, tokenizer=word_tokenize, sentences=True):
    """
    Generator to feed sentences to the dov2vec model.
    """
    i = 0
    p = Progress()
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                i += 1
                p.print_progress(i/n)

                # We do minimal pre-processing here so the model can learn
                # punctuation
                line = line.lower()

                if sentences:
                    for sent in sent_tokenize(line):
                        tokens = tokenizer(sent)
                        yield LabeledSentence(tokens, ['SENT_{}'.format(i)])
                else:
                    tokens = tokenizer(line)
                    yield LabeledSentence(tokens, ['SENT_{}'.format(i)])
