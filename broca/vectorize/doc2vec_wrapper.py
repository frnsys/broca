"""
Doc2Vec wrapper that handles the more idiosyncratic aspects of implementing Gensim's Doc2Vec 
and also implements online testing as described in ...


"""

import numpy as np 
import gensim
from gensim.models.doc2vec import Doc2Vec, LabeledSentence, LabeledLineSentence
import re
import string

class Doc2Vec_Wrapper(Vectorizer):

    def __init__(self, size=300, window=8, min_count=2, workers=8):
        self.is_trained= False
        self.model = None

    def vectorize(self, docs):
        sentences = [self.gen_sentence(item) for item in docs]



    def update(self, train=False):
        n_sentences = self.add_new_labels(sentences)

        # add new rows to self.model.syn0
        n = self.model.syn0.shape[0]
        self.model.syn0 = np.vstack((
            self.model.syn0,
            np.empty((n_sentences, self.model.layer1_size), dtype=np.float32)
        ))

        for i in xrange(n, n + n_sentences):
            np.random.seed(
                np.uint32(self.model.hashfxn(self.model.index2word[i] + str(self.model.seed))))
            a = (np.random.rand(self.model.layer1_size) - 0.5) / self.model.layer1_size
            self.model.syn0[i] = a

        # Set self.model.train_words to False and self.model.train_labels to True
        self.model.train_words = False
        self.model.train_lbls = True

        # train
        self.model.train(sentences)

    def train(self, docs):
        if self.is_trained:
            ## online training 




        else:
            train_sentences = [self.gen_sentence(item) for item in docs]
            self.model = Doc2Vec(train_sentences, size=300, window=8, min_count=2, workers=8)
            ## train from scratch
            self.is_trained = True

    def add_new_labels(self, sentences):
        sentence_no = -1
        total_words = 0
        vocab = self.model.vocab
        model_sentence_n = len([l for l in vocab if l.startswith("SENT")])
        n_sentences = 0
        for sentence_no, sentence in enumerate(sentences):
            sentence_length = len(sentence.words)
            for label in sentence.labels:
                total_words += 1
                if label in vocab:
                    vocab[label].count += sentence_length
                else:
                    vocab[label] = gensim.models.word2vec.Vocab(
                        count=sentence_length)

                    vocab[label].index = len(self.model.vocab) - 1
                    vocab[label].code = [0]
                    vocab[label].sample_probability = 1.
                    self.model.index2word.append(label)
                    n_sentences += 1
                    
        return n_sentences

    def process(self, input):
        input = re.sub("<[^>]*>", " ", input) 
        punct = list(string.punctuation)
        for symbol in punct:
            input = input.replace(symbol, " %s " % symbol)
        input = filter(lambda x: x != u'', input.lower().split(' '))
        return input

    def gen_sentence(self, assetid_body_tuple):
        '''
        assetid_body_tuple: 
            type tuple  
            param (assetid, bodytext) pair 
        '''
        asset_id, body = assetid_body_tuple
        text = self.process(body)
        sentence = LabeledSentence(text, labels=['DOC_%s' % asset_id])
        return sentence

