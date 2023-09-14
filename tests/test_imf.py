import unittest
import sys
import os
import torch

sys.path.insert(0, os.path.split(os.path.split(__file__)[0])[0])
import lighthouse as lh


class TestIMF(unittest.TestCase):

    def test_weights(self):

        N = 100000
        shift = (100 - 0.08)/N 
        masses = torch.linspace(0.08 + shift/2, 100 - shift/2, N)
        dm = masses[1] - masses[0]

        for imf in lh.initial_mass_function.Initial_Mass_Function.__subclasses__():

            weights = imf().get_weight(masses)

            self.assertAlmostEqual(torch.sum(weights*dm).item(), 1, 3, "Check your IMF normalization!!!")

    def test_continuity(self):

        N = 1000000
        shift = (100 - 0.08)/N 
        masses = torch.linspace(0.08 + shift/2, 100 - shift/2, N)

        for imf in lh.initial_mass_function.Initial_Mass_Function.__subclasses__():

            print(imf)

            weights = imf().get_weight(masses)

            delta_weight = weights[1:] - weights[:-1]

            print(delta_weight.abs()/weights[1:])

            self.assertTrue(torch.all(delta_weight.abs()/weights[1:] < 1e-3), "Check your piecewise functions!!")

class TestKroupa(unittest.TestCase):


    def test_kroupa(self):
        

        K = lh.initial_mass_function.Kroupa()

        weight = K.get_imf(torch.tensor(30., dtype = torch.float64))

        self.assertAlmostEqual(weight.detach().cpu().numpy(), 5.1374e-5, 3, "Kroupa IMF test value not equal")
