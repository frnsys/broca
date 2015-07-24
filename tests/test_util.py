import unittest
import numpy as np
from broca.common import util


class UtilTest(unittest.TestCase):
    def test_sim_mat(self):
        def sim_func(a, b):
            return a * b

        items = [1,2,3,4]
        expected = np.array([
            [1., 2., 3., 4.],
            [2., 1., 6., 8.],
            [3., 6., 1., 12.],
            [4., 8., 12., 1.]
        ])

        sim_mat = util.build_sim_mat(items, sim_func)
        np.testing.assert_array_equal(sim_mat, expected)

