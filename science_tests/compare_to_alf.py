import unittest
import sys
import os
import torch

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.split(os.path.split(__file__)[0])[0])
import forsed


if __name__=='__main__':


    S = forsed.SSP.Basic_SSP(
        forsed.isochrone.MIST(),
        forsed.initial_mass_function.Kroupa(),
        forsed.stellar_atmosphere_spectrum.PolynomialEvaluator(),
    )

    ssp = S.forward(torch.tensor(0., dtype = torch.float64), torch.tensor(10., dtype = torch.float64), torch.tensor([1.3, 2.3, 2.7]))

    plt.plot(S.sas.wavelength, ssp)
    plt.show()