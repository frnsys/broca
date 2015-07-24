import unittest
from broca.tokenize import keyword, util, Lemma


class KeywordTokenizeTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            'This cat dog is running happy.',
            'This cat dog runs sad.'
        ]

    def test_overkill(self):
        expected_t_docs = [
            ['run', 'happy', 'cat dog'],
            ['run', 'sad', 'cat dog']
        ]
        t_docs = keyword.Overkill(prune=True, lemmatize=True).tokenize(self.docs)
        self.assertEqual(t_docs, expected_t_docs)

    def test_rake(self):
        expected_t_docs = [
            ['cat dog', 'running happy'],
            ['cat dog runs sad']
        ]
        t_docs = keyword.RAKE().tokenize(self.docs)
        self.assertEqual(t_docs, expected_t_docs)

    def test_apriori(self):
        expected_t_docs = [
            ['cat dog'],
            ['cat dog']
        ]
        t_docs = keyword.Apriori().tokenize(self.docs)
        self.assertEqual(t_docs, expected_t_docs)

    def test_pos(self):
        expected_t_docs = [
            ['cat dog'],
            ['cat dog']
        ]
        t_docs = keyword.POS().tokenize(self.docs)
        self.assertEqual(t_docs, expected_t_docs)


class TokenizeTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            'This cat dog is running happy.',
            'This cat dog runs sad.'
        ]

    def test_lemma(self):
        expected_t_docs = [
            ['cat', 'dog', 'run', 'happy', '.'],
            ['cat', 'dog', 'run', 'sad', '.']
        ]
        t_docs = Lemma().tokenize(self.docs)
        self.assertEqual(t_docs, expected_t_docs)

    def test_prune(self):
        t_docs = [
            ['cat', 'cat dog', 'happy', 'dog', 'dog'],
            ['cat', 'cat dog', 'sad']
        ]
        expected_t_docs = [
            ['cat dog', 'happy', 'dog', 'dog'],
            ['cat dog', 'sad']
        ]
        t_docs = util.prune(t_docs)
        self.assertEqual(t_docs, expected_t_docs)
