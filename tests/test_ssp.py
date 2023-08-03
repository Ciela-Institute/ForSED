import unittest
import sys
import os
import torch

sys.path.insert(0, os.path.split(os.path.split(__file__)[0])[0])
import lighthouse as lh

class TestBasicSSP(unittest.TestCase):
    def test_basicSSP(self):

        S = lh.SSP.Basic_SSP(
            lh.isochrone.MIST(),
            lh.initial_mass_function.Kroupa(),
            lh.stellar_atmosphere_spectrum.PolynomialEvaluator(),
        )

        ssp = S.forward(torch.tensor(0., dtype = torch.float64), torch.tensor(8., dtype = torch.float64), torch.tensor([1.3, 2.3, 2.7]))

        self.assertTrue(len(ssp) == 10565, "this SSP should have a length of 10565")
