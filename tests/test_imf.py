import unittest
import sys
import os
import torch

sys.path.insert(0, os.path.split(os.path.split(__file__)[0])[0])
import forsed

class TestKroupa(unittest.TestCase):
    def test_kroupa(self):

        K = forsed.initial_mass_function.Kroupa()

        weight = K.get_weight(torch.tensor(30., dtype = torch.float64), torch.tensor([1.3, 2.3, 2.7]))

        self.assertAlmostEqual(weight.detach().cpu().numpy(), 5.1374e-5, 3, "Kroupa IMF test value not equal")
