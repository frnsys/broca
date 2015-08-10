import unittest
from broca.tokenize import keyword, util, LemmaTokenizer


class KeywordTokenizeTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            'This cat dog is running happy.',
            'This cat dog runs sad.'
        ]

    def test_overkill(self):
        expected_t_docs = [
            ['cat dog', 'run', 'happy'],
            ['cat dog', 'run', 'sad']
        ]
        t_docs = keyword.OverkillTokenizer(lemmatize=True,
                                           min_count=1,
                                           threshold=0.1).tokenize(self.docs)
        self.assertEqual(t_docs, expected_t_docs)

    def test_rake(self):
        expected_t_docs = [
            ['cat dog', 'running happy'],
            ['cat dog runs sad']
        ]
        t_docs = keyword.RAKETokenizer().tokenize(self.docs)

        # Order not necessarily preserved
        for i, output in enumerate(t_docs):
            self.assertEqual(set(output), set(expected_t_docs[i]))

    def test_apriori(self):
        expected_t_docs = [
            ['cat dog'],
            ['cat dog']
        ]
        t_docs = keyword.AprioriTokenizer().tokenize(self.docs)
        self.assertEqual(t_docs, expected_t_docs)

    def test_pos(self):
        expected_t_docs = [
            ['cat dog'],
            ['cat dog']
        ]
        t_docs = keyword.POSTokenizer().tokenize(self.docs)
        self.assertEqual(t_docs, expected_t_docs)

    def test_overkill_parallel(self):
        expected_t_docs = [
            ['cat dog', 'run', 'happy'],
            ['cat dog', 'run', 'sad']
        ]
        t_docs = keyword.OverkillTokenizer(lemmatize=True,
                                           min_count=1,
                                           threshold=0.1,
                                           n_jobs=2).tokenize(self.docs)
        self.assertEqual(t_docs, expected_t_docs)

    def test_rake_parallel(self):
        expected_t_docs = [
            ['cat dog', 'running happy'],
            ['cat dog runs sad']
        ]
        t_docs = keyword.RAKETokenizer(n_jobs=-1).tokenize(self.docs)

        # Order not necessarily preserved
        for i, output in enumerate(t_docs):
            self.assertEqual(set(output), set(expected_t_docs[i]))


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
        t_docs = LemmaTokenizer().tokenize(self.docs)
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
