import unittest
import sys
import os
import torch

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

sys.path.insert(0, os.path.split(os.path.split(__file__)[0])[0])
import forsed


def grab_alf_ssp(age=11.0, mcut=0.08, Z=+0.0): 

    # data_path = '/Users/alexa/ForSED/forsed/data/alf_data/'
    # fname     = 'VCJ_v8_mcut0.08_t11.0_Zp0.0.ssp.imf_varydoublex.s100'

    # ssp_grid = pd.read_csv(data_path + fname, delim_whitespace=True)

    fname = '/Users/alexa/Documents/Create_SSPs/CheckModels/vcj_cc8_Zp0.00_t13.5.ssp'

    ssp_grid = pd.read_csv(fname, delim_whitespace=True,
                            header=0)


    return ssp_grid

if __name__=='__main__':


    alf_ssps = grab_alf_ssp()

    S = forsed.SSP.Basic_SSP(
        forsed.isochrone.MIST(),
        forsed.initial_mass_function.Kroupa(),
        forsed.stellar_atmosphere_spectrum.PolynomialEvaluator(),
    )

    ssp = S.forward(torch.tensor(0., dtype = torch.float64), torch.tensor(13.5, dtype = torch.float64), torch.tensor([1.3, 2.3, 2.7]))

    plt.plot(alf_ssps['wavelength']*1e-4, alf_ssps['kroupa'], color='k')

    plt.plot(S.sas.wavelength, ssp)
    plt.show()


    print(ssp.shape)
    print(alf_ssps.shape)