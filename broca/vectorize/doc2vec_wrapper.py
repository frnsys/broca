"""
Doc2Vec wrapper that handles the more idiosyncratic aspects of implementing Gensim's Doc2Vec 
and also implements online testing as described in ...

"""

import numpy as np 
import gensim
from gensim.models.doc2vec import Doc2Vec, LabeledSentence, LabeledLineSentence
from ..broca.pipeline import Pipe 
import re
import string
import Vectorizer


class Doc2Vec_Wrapper(Vectorizer):

    input = Pipe.type.assetid_docs
    output = Pipe.type.assetid_vecs

    def __init__(self, size=300, window=8, min_count=2, workers=8, path_to_model=None, stream_train=False):

        '''
        Initializes the Doc2Vec_Wrapper class. 

        Args:
            size (int): Specifies the size of the feature-vector. Defaults to 300
            window (int): Specifies the size of the context window from which the feature vector is learned
            min_count (int): Specifices the minimum number of instances of each word that is saved in the model
            workers (int): number of parallel processes
            path_to_model (str): Specifies model on disk 
            stream_train (bool): If true, update word vectors with new sentences. If false, just get doc vecs
        '''

        self.stream_train=stream_train

        self.is_trained = False
        self.model = None

        ## if a path is passed, try to load from disk. Otherwise, retrain anyway
        if path_to_model:
            try:
                self.is_trained = True
                self.model = Doc2Vec.load(path_to_model)
            except:
                pass

        ## params for Doc2Vec 
        self.size = size ## size of the vector
        self.window = window ## size of the context window
        self.min_count = min_count ## minimum count of vocab to store in binary tree
        self.workers = workers ## number of parallel processes == number of cores on the computer


    def __call__(self, docs):
        vecs = self.vectorize(docs)
        return vecs


    def vectorize( self, docs ):
        '''
        Returns the feature vectors for a set of docs. If model is not already be trained, 
        then self.train() is called.

        Args:
            docs (dict or list of tuples): asset_id, body_text of documents
            you wish to featurize.
        '''

        if type(docs) == dict:
            docs = docs.items()

        if self.model == None:
            self.train(docs)

        asset_id2vector = {}

        unfound = []
        for item in docs:
            ## iterate through the items in docs and check if any are already in the model.
            asset_id, _ = item
            label = 'DOC_' + str(asset_id)
            if label in self.model:
                asset_id2vector.update({asset_id: self.model['DOC_' + str(asset_id)]})
            else:
                unfound.append(item)

        if len(unfound) > 0:
            ## for all assets not in the model, update the model and then get their sentence vectors.
            sentences = [self._gen_sentence(item) for item in unfound]
            self.update_model(sentences, train=self.stream_train)
            asset_id2vector.update({item[0]: self.model['DOC_' + str(item[0])] for item in unfound})

        return asset_id2vector


    def train(self, docs, retrain=False):
        '''
        Train Doc2Vec on a series of docs. Train from scratch or update.

        input:
            docs: list of tuples (assetid, body_text) or dictionary {assetid : body_text}
            retrain: boolean, retrain from scratch or update model

        output: saves model in class to self.model   
        '''

        if type(docs) == dict:
            docs = docs.items()

        train_sentences = [self._gen_sentence(item) for item in docs]
        if (self.is_trained) and (retrain == False): 
            ## online training 
            self.update_model(sentences, update_labels_bool=False)

        else: 
            ## train from scratch
            self.model = Doc2Vec(train_sentences, size=self.size, window=self.window, min_count=self.min_count, workers=self.workers)
            self.is_trained = True

        return 'done training'


    def save_model(self, path):
        '''
        Saves the model to a path.

        Args:
            path (str): relative or absolute path to save the module
        '''
        self.model.save(path)


    def update_model(self, sentences, update_labels_bool):
        '''
        takes a list of sentenes and updates an existing model. Vectors will be 
        callable through self.model[label]

        update_labels_bool: boolean that says whether to train the model (self.model.train_words = True)
        or simply to get vectors for the documents (self.model.train_words = False)

            self.vectorize should not train the model further
            self.train should if model already exists

        '''

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
        self.model.train_words = update_labels_bool
        self.model.train_lbls = True

        # train
        self.model.train(sentences)
        return 


    def _process(self, input):
        '''
        Takes in html-mixed body text as a string and returns a list of strings,
        lower case and with punctuation given spacing. 

        Called by self._gen_sentence()

        Args:
            inpnut (string): body text
        '''

        input = re.sub("<[^>]*>", " ", input) 
        punct = list(string.punctuation)
        for symbol in punct:
            input = input.replace(symbol, " %s " % symbol)
        input = filter(lambda x: x != u'', input.lower().split(' '))
        return input


    def _gen_sentence(self, assetid_body_tuple):
        '''
        Takes an assetid_body_tuple and returns a Doc2Vec LabeledSentence 

        Args:
            assetid_body_tuple (tuple): (assetid, bodytext) pair 
        '''
        asset_id, body = assetid_body_tuple
        text = self.process(body)
        sentence = LabeledSentence(text, labels=['DOC_%s' % str(asset_id)])
        return sentence


    def _add_new_labels(self, sentences):
        '''
        Adds new sentences to the internal indexing of the model.

        Args: 
            sentences (list): LabeledSentences for each doc to be added

        Returns:
            int: number of sentences added to the model

        '''
        sentence_no = -1
        total_words = 0
        vocab = self.model.vocab
        model_sentence_n = len([l for l in vocab if l.startswith("DOC_")])
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

