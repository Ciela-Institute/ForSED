import unittest
import sys
import os
import torch

sys.path.insert(0, os.path.split(os.path.split(__file__)[0])[0])
import forsed

class TestPolyEval(unittest.TestCase):
    def test_polynomialevaluator(self):

        P = forsed.stellar_atmosphere_spectrum.PolynomialEvaluator()
        spec = P.get_spectrum(torch.tensor(3000., dtype = torch.float64), torch.tensor(4.,dtype = torch.float64), torch.tensor(0., dtype = torch.float64))

        self.assertTrue(len(spec) == 10565, "This spectrum should have a length of 10565 but instead is {len(spec)}")
