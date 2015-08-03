import unittest
from broca.preprocess import BasicCleaner, HTMLCleaner


class PreProcessTests(unittest.TestCase):
    def test_clean(self):
        doc = '''
        Goats are like mushrooms. If you shoot a duck, I'm scared of toasters. My site's are https://google.com.
        '''
        expected_doc = '''
        goats are like mushrooms if you shoot a duck im scared of toasters my site are
        '''

        doc = BasicCleaner().preprocess([doc])[0]
        self.assertEqual(doc, expected_doc.strip())

    def test_html_clean(self):
        doc = '''
        <html>goats are like <b>mushrooms</b> if you shoot a duck <em>im scared of toasters</em> my site are<div></div></html>
        '''
        expected_doc = '''
        goats are like mushrooms if you shoot a duck im scared of toasters my site are
        '''

        doc = HTMLCleaner().preprocess([doc])[0]
        self.assertEqual(doc, expected_doc.strip())
