import unittest
from broca.visualize.highlight import markup_terms
class MarkupTests(unittest.TestCase):
    def test_highlight(self):
        doc = 'This cat dog is happy.'
        t_doc = ['happy', 'cat dog']
        expected = 'This <span class="highlight" data-term="cat dog">cat dog</span> is <span class="highlight" data-term="happy">happy</span>.'
        highlighted = markup_terms([doc], [t_doc])
        self.assertEqual(highlighted, [expected])
