"""
An example for building an IDF dictionary off of some documents.
"""

from time import time
from glob import glob
from broca import Pipeline
from broca.preprocess import HTMLCleaner, BasicCleaner
from broca.tokenize.keyword import OverkillTokenizer
from broca.knowledge.idf import train_idf
from broca.knowledge.util import files_stream


s = time()
print('Loading documents...')
files = glob('bodies/*.txt')
docs = [d for d in files_stream(files)]

tkn = OverkillTokenizer(n_jobs=-1)

pipeline = Pipeline(HTMLCleaner(n_jobs=-1), BasicCleaner(n_jobs=-1), tkn, refresh=True)

print('Computing pipeline...')
tokens = pipeline(docs)

print('Training IDF...')
train_idf(tokens, out='nyt_idf.json')

print('Took {:.2f}s'.format(time() - s))

tkn.bigram.save('nyt.bigram')
tkn.trigram.save('nyt.trigram')