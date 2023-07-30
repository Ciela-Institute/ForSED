import unittest
import sys
import os
import torch

import matplotlib.pyplot as plt

import pandas as pd

sys.path.insert(0, os.path.split(os.path.split(__file__)[0])[0])
import forsed


def grab_alf_ssp(): 

    data_path = '/Users/alexa/ForSED/forsed/data/alf_data/'
    fname     = 'VCJ_v8_mcut0.08_t11.0_Zp0.0.ssp.imf_varydoublex.s100'

    grid = pd.read_csv(data_path + fname, delim_whitespace=True)

    wavelength = grid.iloc[:, 0]
    ssp_grid   = grid.iloc[:, 1:]

    plt.plot(wavelength, ssp_grid.iloc[:,1])
    plt.show()

if __name__=='__main__':


    grab_alf_ssp()

    # S = forsed.SSP.Basic_SSP(
    #     forsed.isochrone.MIST(),
    #     forsed.initial_mass_function.Kroupa(),
    #     forsed.stellar_atmosphere_spectrum.PolynomialEvaluator(),
    # )

    # ssp = S.forward(torch.tensor(0., dtype = torch.float64), torch.tensor(10., dtype = torch.float64), torch.tensor([1.3, 2.3, 2.7]))

    # plt.plot(S.sas.wavelength, ssp)
    # plt.show()