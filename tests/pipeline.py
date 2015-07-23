import unittest
from broca.pipeline import Pipeline
from broca.preprocess import Cleaner
from broca.tokenize.keyword import Overkill


class PipelineTests(unittest.TestCase):
    def test_pipeline(self):
        docs = [
        '''
        In the Before-Time, there was only the Vast Empty. No space nor time did continue, all the dimensions were held in their nonexistence. The Great Nicolas Cage thus was sprung into existence through His very will, and saw all nothing, and was displeased.
        ''',
        '''
        As a Galactic Ocean floated by, Nicolas Cage reached out His hand and grasped it. He looked at it with His glorious eyes and instantaneously began to stretch it and bend it. He found amusement in the handling of these Sacred Galactic Seas. And so, Cage reached out His mighty hand and took another Ocean into the sacred warmth of His mighty palms.
        '''
        ]
        expected = [
            ['time', 'vast', 'empty', 'space', 'time', 'continue', 'dimension', 'hold', 'nonexistence', 'great', 'spring', 'displeased', 'nicolas cage'],
            ['galactic', 'ocean', 'float', 'reach', 'hand', 'grasp', 'look', 'glorious', 'eye', 'instantaneously', 'begin', 'stretch', 'bend', 'find', 'amusement', 'handling', 'sacred', 'galactic', 'sea', 'reach', 'mighty', 'hand', 'ocean', 'sacred', 'warmth', 'mighty', 'palm', 'cage reach', 'nicolas cage']
        ]

        pipeline = Pipeline(Cleaner(), Overkill())
        output = pipeline(docs)
        for o, e in zip(output, expected):
            self.assertEqual(set(o), set(e))

    def test_incompatible_pipeline(self):
        self.assertRaises(Exception, Pipeline, Overkill(), Cleaner())
