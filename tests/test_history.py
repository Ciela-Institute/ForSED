import unittest
import sys
import os

import torch

sys.path.insert(0, os.path.split(os.path.split(__file__)[0])[0])
import lighthouse as lh

class TestHistory(unittest.TestCase):

    def test_constant(self):

        constant_value = torch.tensor(5.)
        H = lh.history.Constant_History(constant_value)

        for t in torch.tensor((0., 1., 5., 10.)):
            self.assertEqual(H(t), constant_value, "Constant history should always return the same value")
            self.assertEqual(H.window(t, t + 1), constant_value, "Constant history should always return the same value")
