import unittest
from broca import Pipe, Pipeline
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

    def test_valid_branching_pipeline_multiout_to_branches(self):
        class A(Pipe):
            input = Pipe.type.a
            output = (Pipe.type.b, Pipe.type.c, Pipe.type.d)

        class B(Pipe):
            input = Pipe.type.b
            output = Pipe.type.b_out

        class C(Pipe):
            input = Pipe.type.c
            output = Pipe.type.c_out

        class D(Pipe):
            input = Pipe.type.d
            output = Pipe.type.d_out

        class E(Pipe):
            input = (Pipe.type.b_out, Pipe.type.c_out, Pipe.type.d_out)
            output = Pipe.type.e

        try:
            Pipeline(
                A(),
                (B(), C(), D()),
                E()
            )
        except Exception:
            self.fail('Valid pipeline raised exception')

    def test_invalid_branching_pipeline_multiout_to_branches(self):
        class A(Pipe):
            input = Pipe.type.a
            # A outputs tuples
            output = (Pipe.type.b, Pipe.type.c, Pipe.type.d)

        class B(Pipe):
            input = Pipe.type.b
            output = Pipe.type.b_out

        class C(Pipe):
            input = Pipe.type.c
            output = Pipe.type.c_out

        class D(Pipe):
            input = Pipe.type.d
            output = Pipe.type.d_out

        class E(Pipe):
            input = (Pipe.type.b_out, Pipe.type.c_out, Pipe.type.d_out)
            output = Pipe.type.e

        # Wrong branch size
        self.assertRaises(Exception, Pipeline, A(), (B(), C()), E())

        # Wrong branch order
        self.assertRaises(Exception, Pipeline, A(), (C(), B(), D()), E())

        # Wrong input type
        class D_(Pipe):
            input = Pipe.type.x
            output = Pipe.type.d_out

        self.assertRaises(Exception, Pipeline, A(), (B(), C(), D_()), E())

        # Wrong output size
        class A_(Pipe):
            input = Pipe.type.a
            output = (Pipe.type.b, Pipe.type.c)

        self.assertRaises(Exception, Pipeline, A_(), (B(), C(), D()), E())

        # Wrong output types
        class A_(Pipe):
            input = Pipe.type.a
            output = (Pipe.type.b, Pipe.type.c, Pipe.type.y)

        self.assertRaises(Exception, Pipeline, A_(), (B(), C(), D()), E())

    def test_valid_branching_pipeline_branches_to_branches(self):
        class A(Pipe):
            input = Pipe.type.a
            # A outputs tuples
            output = (Pipe.type.b, Pipe.type.c, Pipe.type.d)

        class B(Pipe):
            input = Pipe.type.b
            output = Pipe.type.b

        class C(Pipe):
            input = Pipe.type.c
            output = Pipe.type.c

        class D(Pipe):
            input = Pipe.type.d
            output = Pipe.type.d

        class E(Pipe):
            input = (Pipe.type.b, Pipe.type.c, Pipe.type.d)
            output = Pipe.type.e

        try:
            Pipeline(
                A(),
                (B(), C(), D()),
                (B(), C(), D()),
                E()
            )
        except Exception:
            self.fail('Valid pipeline raised exception')

    def test_invalid_branching_pipeline_branches_to_branches(self):
        class A(Pipe):
            input = Pipe.type.a
            # A outputs tuples
            output = (Pipe.type.b, Pipe.type.c, Pipe.type.d)

        class B(Pipe):
            input = Pipe.type.b
            output = Pipe.type.x

        class C(Pipe):
            input = Pipe.type.c
            output = Pipe.type.x

        class D(Pipe):
            input = Pipe.type.d
            output = Pipe.type.x

        class E(Pipe):
            input = (Pipe.type.b, Pipe.type.c, Pipe.type.d)
            output = Pipe.type.e

        self.assertRaises(Exception, Pipeline, A(), (B(), C(), D()), (B(), C(), D()), E())

    def test_valid_branching_pipeline_one_output_to_branches(self):
        class A(Pipe):
            input = Pipe.type.a
            # A does not output tuples
            output = Pipe.type.x

        class B(Pipe):
            input = Pipe.type.x
            output = Pipe.type.b_out

        class C(Pipe):
            input = Pipe.type.x
            output = Pipe.type.c_out

        class D(Pipe):
            input = Pipe.type.x
            output = Pipe.type.d_out

        class E(Pipe):
            input = (Pipe.type.b_out, Pipe.type.c_out, Pipe.type.d_out)
            output = Pipe.type.e

        try:
            Pipeline(
                A(),
                (B(), C(), D()),
                E()
            )
        except Exception:
            self.fail('Valid pipeline raised exception')

    def test_invalid_branching_pipeline_one_output_to_branches(self):
        class A(Pipe):
            input = Pipe.type.a
            # A does not output tuples
            output = Pipe.type.x

        class B(Pipe):
            input = Pipe.type.x
            output = Pipe.type.b_out

        class C(Pipe):
            input = Pipe.type.x
            output = Pipe.type.c_out

        class D(Pipe):
            input = Pipe.type.y
            output = Pipe.type.d_out

        class E(Pipe):
            input = (Pipe.type.b_out, Pipe.type.c_out, Pipe.type.d_out)
            output = Pipe.type.e

        self.assertRaises(Exception, Pipeline, A(), (B(), C(), D()), E())

    def test_invalid_branching_pipeline_reduce_pipe(self):
        class A(Pipe):
            input = Pipe.type.a
            # A does not output tuples
            output = Pipe.type.x

        class B(Pipe):
            input = Pipe.type.x
            output = Pipe.type.b_out

        class C(Pipe):
            input = Pipe.type.x
            output = Pipe.type.c_out

        class D(Pipe):
            input = Pipe.type.x
            output = Pipe.type.d_out

        class E(Pipe):
            input = (Pipe.type.b_out, Pipe.type.c_out, Pipe.type.y)
            output = Pipe.type.e

        self.assertRaises(Exception, Pipeline, A(), (B(), C(), D()), E())

    def test_valid_branching_pipeline_start_with_branches(self):
        class B(Pipe):
            input = Pipe.type.x
            output = Pipe.type.b_out

        class C(Pipe):
            input = Pipe.type.x
            output = Pipe.type.c_out

        class D(Pipe):
            input = Pipe.type.x
            output = Pipe.type.d_out

        class E(Pipe):
            input = (Pipe.type.b_out, Pipe.type.c_out, Pipe.type.d_out)
            output = Pipe.type.e

        try:
            Pipeline(
                (B(), C(), D()),
                E()
            )
        except Exception:
            self.fail('Valid pipeline raised exception')

    def test_valid_branching_pipeline_end_with_branches(self):
        class A(Pipe):
            input = Pipe.type.a
            # A does not output tuples
            output = Pipe.type.x

        class B(Pipe):
            input = Pipe.type.x
            output = Pipe.type.b_out

        class C(Pipe):
            input = Pipe.type.x
            output = Pipe.type.c_out

        class D(Pipe):
            input = Pipe.type.x
            output = Pipe.type.d_out

        try:
            Pipeline(
                A(),
                (B(), C(), D()),
            )
        except Exception:
            self.fail('Valid pipeline raised exception')

    def test_branching_pipeline(self):
        class A(Pipe):
            input = Pipe.type.vals
            output = Pipe.type.vals
            def __call__(self, vals):
                return [v+1 for v in vals]

        class B(Pipe):
            input = Pipe.type.vals
            output = Pipe.type.vals
            def __call__(self, vals):
                return [v+2 for v in vals]

        class C(Pipe):
            input = Pipe.type.vals
            output = Pipe.type.vals
            def __call__(self, vals):
                return [v+3 for v in vals]

        class D(Pipe):
            input = Pipe.type.vals
            output = Pipe.type.vals
            def __call__(self, vals):
                return [v+4 for v in vals]

        class E(Pipe):
            input = (Pipe.type.vals, Pipe.type.vals, Pipe.type.vals)
            output = Pipe.type.vals
            def __call__(self, vals1, vals2, vals3):
                return [sum([v1,v2,v3]) for v1,v2,v3 in zip(vals1,vals2,vals3)]

        p = Pipeline(
            A(),
            (B(), C(), D()),
            (B(), C(), D()),
            E()
        )

        out = p([1,2,3,4])
        self.assertEqual(out, [24,27,30,33])
