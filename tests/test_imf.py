import unittest
import sys
import os
import torch

sys.path.insert(0, os.path.split(os.path.split(__file__)[0])[0])
import lighthouse as lh


class TestIMFWeights(unittest.TestCase):

    def test_imf_weights(self):

        masses = torch.linspace(0.08, 100, 1000)
        dm = masses[1] - masses[0]

        for imf in lh.initial_mass_function.Initial_Mass_Function.__subclasses__():

            weights = imf().get_weight(masses)

            self.assertAlmostEqual(torch.sum(weights*dm), 1, 3, "Check your IMF normalization!!!")


class TestKroupa(unittest.TestCase):
    def test_kroupa(self):
        

        K = lh.initial_mass_function.Kroupa()

        weight = K.get_imf(torch.tensor(30., dtype = torch.float64))

        self.assertAlmostEqual(weight.detach().cpu().numpy(), 5.1374e-5, 3, "Kroupa IMF test value not equal")
