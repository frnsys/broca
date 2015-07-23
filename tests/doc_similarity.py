import unittest
from broca.similarity import doc as doc_sim


class DocSimilarityTests(unittest.TestCase):
    docs = [
        '''
        Here is a celebration of the sandwich's diversity in the United States,
        an attempt to bring order to the wild multiplicity of its forms.
        ''',
        '''
        But first: What is a sandwich? The United States Department of Agriculture
        declares: "Product must contain at least 35 percent cooked meat and no more
        than 50 percent bread." But a sandwich does not require meat! Merriam-Webster
        is slightly more helpful: "two or more slices of bread or a split roll having a
        filling in between."
        ''',
        '''
        For the purposes of this field guide, we have laid down parameters. A hamburger
        is a marvelous sandwich, but it is one deserving of its own guide. The same holds
        for hot dogs, and for tacos and burritos, which in 2006, in the case known as
        Panera v. Qdoba, a Massachusetts judge declared were not sandwiches at all.
        Open-faced sandwiches are not sandwiches. Gyros and shawarmas are not sandwiches.
        The bread that encases them is neither split nor hinged, but wrapped.
        '''
    ]

    def test_wikipedia(self):
        m = doc_sim.WikipediaSimilarity()

        sims = m.sim_mat(self.docs)
        print(sims)
