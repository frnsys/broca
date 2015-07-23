import unittest
from broca.similarity import term as term_sim


class TermSimilarityTests(unittest.TestCase):
    def test_wikipedia(self):
        terms = ['renewable energy', 'alternative energy', 'socrates', 'plato']

        m = term_sim.WikipediaSimilarity(terms)

        self.assertTrue(m['renewable energy', 'alternative energy'] > 0.5)
        self.assertTrue(m['socrates', 'plato'] > 0.5)
        self.assertTrue(m['renewable energy', 'an unknown term'] == 0.)
