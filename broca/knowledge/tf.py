import json
from functools import partial
from collections import defaultdict
from sup.parallel import parallelize
from nltk.tokenize import word_tokenize
from broca.knowledge.util import merge, split_file, doc_stream


def train_tf(paths, out='data/tf.json', tokenizer=word_tokenize, preprocessor=None, **kwargs):
    """
    Train a map of term frequencies on a list of files (parallelized).
    """
    print('Preparing files')
    args = []
    method = kwargs.get('method', 'keyword')
    for path in paths:
        args += [(file, method) for file in split_file(path, chunk_size=5000)]

    # Leave a generous timeout in case the
    # phrases model needs to be loaded.
    print('Counting terms...')
    p_count_tf = partial(count_tf, tokenizer=tokenizer, preprocessor=preprocessor)
    results = parallelize(p_count_tf, args, timeout=360)

    print('Merging...')
    tf = merge(results)

    with open(out, 'w') as f:
        json.dump(tf, f)


def count_tf(path, tokenizer, preprocessor):
    """
    Count term frequencies for a single file.
    """
    tf = defaultdict(int)
    for tokens in doc_stream(path, tokenizer=tokenizer, preprocessor=preprocessor):
        for token in tokens:
            tf[token] += 1
    return tf
