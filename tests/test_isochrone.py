import unittest
import sys
import os
import torch

sys.path.insert(0, os.path.split(os.path.split(__file__)[0])[0])
import forsed

class TestMIST(unittest.TestCase):
    def test_mist(self):

        I = forsed.isochrone.MIST()
        iso = I.get_isochrone(torch.tensor(0., dtype = torch.float64), torch.tensor(8.,dtype = torch.float64))

        for key in ["log_g", "Teff", "initial_mass"]:
            self.assertTrue(key in iso, f"Isochrone should be dictionary with this key: {key}")
            self.assertTrue(len(iso[key]) == 1258, f"This Isochrone should have a length of 1258, but instead is {len(iso[key])}")
