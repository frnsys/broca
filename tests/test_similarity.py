import unittest
from broca import Pipe, Pipeline, IdentityPipe
from broca.similarity import doc as doc_sim
from broca.similarity import term as term_sim
from broca.tokenize.keyword import RAKETokenizer
from broca.entity import Entities


class DocSimilarityTests(unittest.TestCase):
    docs = [
        '''
        Here is a celebration of the sandwich's diversity in the United States.
        ''',
        '''
        A hamburger is a marvelous sandwich, but it is one deserving of its own guide.
        ''',
        '''
        A hamburger is a marvelous sandwich, but it is one deserving of its own guide.
        '''
    ]

    def test_wikipedia(self):
        # TO DO not sure if this is implemented correctly; not getting 1. sim
        # for identical documents...
        p = Pipeline(
            (IdentityPipe(Pipe.type.docs), RAKETokenizer()),
            doc_sim.WikipediaSimilarity()
        )
        sims = p(self.docs)
        self.assertEqual(sims.shape, (3,3))

    def test_entkey(self):
        class FauxIDF():
            def __getitem__(self, key):
                return 1.

        p = Pipeline(
            (RAKETokenizer(), Entities()),
            doc_sim.EntKeySimilarity(idf=FauxIDF())
        )

        sims = p(self.docs)
        self.assertEqual(sims.shape, (3,3))


class TermSimilarityTests(unittest.TestCase):
    def test_wikipedia(self):
        terms = ['renewable energy', 'alternative energy', 'socrates', 'plato']

        m = term_sim.WikipediaSimilarity(terms)

        self.assertTrue(m['renewable energy', 'alternative energy'] > 0.5)
        self.assertTrue(m['socrates', 'plato'] > 0.5)
        self.assertTrue(m['renewable energy', 'an unknown term'] == 0.)
