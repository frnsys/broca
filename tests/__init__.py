import unittest
#from broca.tokenize import keyword


#class KeywordTokenizeTests(unittest.TestCase):
    #def setUp(self):
        #pass

    #def test_prune(self):
        #t_docs = [
            #['cat', 'cat dog', 'happy', 'dog', 'dog'],
            #['cat', 'cat dog', 'sad']
        #]
        #expected_t_docs = [
            #['cat dog', 'happy', 'dog', 'dog'],
            #['cat dog', 'sad']
        #]
        #terms = {t for ts in t_docs for t in ts}
        #t_docs, redundant = keyword.prune(terms, t_docs)

        #self.assertEqual(redundant, {'cat'})
        #self.assertEqual(t_docs, expected_t_docs)

    #def test_keyword_tokenizes(self):
        #docs = [
            #'This cat dog is running happy.',
            #'This cat dog is ran sad.'
        #]
        #expected_t_docs = [
            #['run', 'happy', 'cat dog'],
            #['run', 'sad', 'cat dog']
        #]
        #t_docs = keyword.keyword_tokenizes(docs)
        #self.assertEqual(t_docs, expected_t_docs)


#from broca.visualize.highlight import markup_terms
#class MarkupTests(unittest.TestCase):
    #def test_highlight(self):
        #doc = 'This cat dog is happy.'
        #t_doc = ['happy', 'cat dog']
        #expected = 'This <span class="highlight" data-term="cat dog">cat dog</span> is <span class="highlight" data-term="happy">happy</span>.'
        #highlighted = markup_terms([doc], [t_doc])
        #self.assertEqual(highlighted, [expected])