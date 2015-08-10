import json
from collections import defaultdict
from broca.common.util import parallel
from broca.knowledge.util import merge


def train_tf(tokens_stream, out='data/tf.json', **kwargs):
    """
    Train a map of term frequencies on a list of files (parallelized).
    """
    print('Counting terms...')
    results = parallel(count_tf, tokens_stream, n_jobs=-1)

    print('Merging...')
    tf = merge(results)

    with open(out, 'w') as f:
        json.dump(tf, f)


def count_tf(tokens_stream):
    """
    Count term frequencies for a single file.
    """
    tf = defaultdict(int)
    for tokens in tokens_stream:
        for token in tokens:
            tf[token] += 1
    return tf
