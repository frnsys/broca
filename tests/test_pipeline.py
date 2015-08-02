import unittest
from broca.pipeline import Pipe, Pipeline
from broca.preprocess import Cleaner, HTMLCleaner
from broca.tokenize.keyword import Overkill, RAKE


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
        '''In the Before-Time, there was only the Vast Empty. No space nor time did continue, all the dimensions were held in their nonexistence. The Great Nicolas Cage thus was sprung into existence through His very will, and saw all nothing, and was displeased.''',
        '''As a Galactic Ocean floated by, Nicolas Cage reached out His hand and grasped it. He looked at it with His glorious eyes and instantaneously began to stretch it and bend it. He found amusement in the handling of these Sacred Galactic Seas. And so, Cage reached out His mighty hand and took another Ocean into the sacred warmth of His mighty palms.'''
        ]

    def test_pipeline(self):
        expected = [
            ['time', 'vast', 'empty', 'space', 'time', 'continue', 'dimension', 'hold', 'nonexistence', 'great', 'spring', 'displeased', 'nicolas cage'],
            ['galactic', 'ocean', 'float', 'reach', 'hand', 'grasp', 'look', 'glorious', 'eye', 'instantaneously', 'begin', 'stretch', 'bend', 'find', 'amusement', 'handling', 'sacred', 'galactic', 'sea', 'reach', 'mighty', 'hand', 'ocean', 'sacred', 'warmth', 'mighty', 'palm', 'cage reach', 'nicolas cage']
        ]
        pipeline = Pipeline(Cleaner(), Overkill())
        output = pipeline(self.docs)
        for o, e in zip(output, expected):
            self.assertEqual(set(o), set(e))

    def test_incompatible_pipeline(self):
        self.assertRaises(Exception, Pipeline, Overkill(), Cleaner())

    def test_multi_pipeline(self):
        pipeline = Pipeline(
            Cleaner(),
            [Overkill(), RAKE()]
        )
        expected = [
            [
                ['vast', 'empty', 'space', 'time', 'continue', 'dimension', 'hold', 'nonexistence', 'great', 'spring', 'displeased', 'nicolas cage'],
                ['galactic', 'ocean', 'float', 'reach', 'hand', 'grasp', 'look', 'glorious', 'eye', 'instantaneously', 'begin', 'stretch', 'bend', 'find', 'amusement', 'handling', 'sacred', 'galactic', 'sea', 'reach', 'mighty', 'hand', 'ocean', 'sacred', 'warmth', 'mighty', 'palm', 'cage reach', 'nicolas cage']
            ],
            [
                ['great nicolas cage', 'vast empty', 'sprung', 'nonexistence', 'dimensions', 'held', 'existence', 'displeased', 'continue', 'time', 'space'],
                ['sacred galactic seas', 'galactic ocean floated', 'nicolas cage reached', 'cage reached', 'sacred warmth', 'glorious eyes', 'mighty palms', 'found amusement', 'instantaneously began', 'mighty hand', 'ocean', 'hand', 'looked', 'stretch', 'grasped', 'handling', 'bend']
            ]
        ]
        outputs = pipeline(self.docs)
        for i, output in enumerate(outputs):
            for o, e in zip(output, expected[i]):
                self.assertEqual(set(o), set(e))

    def test_nested_pipeline(self):
        docs = ['<div>{}</div>'.format(d) for d in self.docs]
        expected = [
            ['time', 'vast', 'empty', 'space', 'time', 'continue', 'dimension', 'hold', 'nonexistence', 'great', 'spring', 'displeased', 'nicolas cage'],
            ['galactic', 'ocean', 'float', 'reach', 'hand', 'grasp', 'look', 'glorious', 'eye', 'instantaneously', 'begin', 'stretch', 'bend', 'find', 'amusement', 'handling', 'sacred', 'galactic', 'sea', 'reach', 'mighty', 'hand', 'ocean', 'sacred', 'warmth', 'mighty', 'palm', 'cage reach', 'nicolas cage']
        ]
        nested_pipeline = Pipeline(HTMLCleaner(), Cleaner())
        pipeline = Pipeline(nested_pipeline, Overkill())
        output = pipeline(docs)
        for o, e in zip(output, expected):
            self.assertEqual(set(o), set(e))

    def test_nested_multipipeline(self):
        docs = ['<div>{}</div>'.format(d) for d in self.docs]
        expected = [
            [
                ['vast', 'empty', 'space', 'time', 'continue', 'dimension', 'hold', 'nonexistence', 'great', 'spring', 'displeased', 'nicolas cage'],
                ['galactic', 'ocean', 'float', 'reach', 'hand', 'grasp', 'look', 'glorious', 'eye', 'instantaneously', 'begin', 'stretch', 'bend', 'find', 'amusement', 'handling', 'sacred', 'galactic', 'sea', 'reach', 'mighty', 'hand', 'ocean', 'sacred', 'warmth', 'mighty', 'palm', 'cage reach', 'nicolas cage']
            ],
            [
                ['great nicolas cage', 'vast empty', 'sprung', 'nonexistence', 'dimensions', 'held', 'existence', 'displeased', 'continue', 'time', 'space'],
                ['sacred galactic seas', 'galactic ocean floated', 'nicolas cage reached', 'cage reached', 'sacred warmth', 'glorious eyes', 'mighty palms', 'found amusement', 'instantaneously began', 'mighty hand', 'ocean', 'hand', 'looked', 'stretch', 'grasped', 'handling', 'bend']
            ]
        ]
        nested_multipipeline = Pipeline(
            Cleaner(),
            [Overkill(), RAKE()]
        )
        pipeline = Pipeline(HTMLCleaner(), nested_multipipeline)
        outputs = pipeline(docs)
        for i, output in enumerate(outputs):
            for o, e in zip(output, expected[i]):
                self.assertEqual(set(o), set(e))

    def test_cryo_diff_pipe_init(self):
        pipeline = Pipeline(
            Cleaner(),
        )
        output1 = pipeline(self.docs)

        pipeline = Pipeline(
            Cleaner(),
        )
        output2 = pipeline(self.docs)
        self.assertEqual(output1, output2)

        # Make sure cryo picks up on differently initialized classes
        pipeline = Pipeline(
            Cleaner(lowercase=False),
        )
        output3 = pipeline(self.docs)
        self.assertNotEqual(output1, output3)

    def test_dynamic_pipe_types(self):
        self.assertEqual(Pipe.type.foo, Pipe.type.foo)
        self.assertEqual(Pipe.type.bar, Pipe.type.bar)
        self.assertNotEqual(Pipe.type.foo, Pipe.type.bar)
