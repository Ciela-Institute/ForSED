import torch
from functools import partial
from time import process_time as time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from .. import utils

class Basic_SSP():

    def __init__(
            self,
            isochrone_model: "Isochrone",
            imf: "Initial_Mass_Function",
            sas: "Stellar_Atmosphere_Spectrum",
    ):

        self.isochrone_grid = isochrone_model
        self.imf = imf
        self.sas = sas

        self.isochrone  = None
        self._tage_normalization = None
        self.highest_mass = None
        self.ms_turnoff_point = None


    def get_ms_turnoff(self, ms_isochrone):

        turnoff = ms_isochrone["Teff"].argmax()

        xx = -1.0
        while xx < 0.0:
            xx = ms_isochrone[turnoff]["log_L"] - ms_isochrone[turnoff-1]["log_L"]
            if xx < 0.0:
                turnoff = turnoff - 1

        self.ms_turnoff_point = turnoff


    def get_imf_weights(self,  initial_stellar_masses) -> torch.Tensor:

        lower_limit = self.imf.lower_limit
        upper_limit = self.imf.upper_limit

        weights = torch.zeros(initial_stellar_masses.shape)
        for i, mass in enumerate(initial_stellar_masses):
            if initial_stellar_masses[i] < lower_limit or initial_stellar_masses[i] > upper_limit:
                print("Bounds of isochrone exceed limits of full IMF")
            if i == 0:
                m1 = lower_limit
            else:
                m1 = initial_stellar_masses[i] - 0.5*(initial_stellar_masses[i] - initial_stellar_masses[i-1])
            if i == len(initial_stellar_masses) - 1:
                m2 = initial_stellar_masses[i]
            else:
                m2 = initial_stellar_masses[i] + 0.5*(initial_stellar_masses[i+1] - initial_stellar_masses[i])

            if m2 < m1:
                print("IMF_WEIGHT WARNING: non-monotonic mass!", m1, m2, m2-m1)
                continue

            if m2 == m1:
                print("m2 == m1")
                continue

            tmp, error =  quad(self.imf.get_imf,
                                m1,
                                m2,
                                args=(False,) )
            weights[i] = tmp

        imf_weight = weights/self.imf.t0_normalization

        return imf_weight



    def test(self, metalicity, Tage) -> torch.Tensor:

        """
        Using FSPS as a guide
        """

        self.isochrone = self.isochrone_grid.get_isochrone(metalicity, Tage)

        log10_imf_weights = torch.log10(self.get_imf_weights(self.isochrone["initial_mass"]))

        return log10_imf_weights


    def forward(self, metalicity, Tage) -> torch.Tensor:

        self.isochrone = self.isochrone_grid.get_isochrone(metalicity, Tage)
        isochrone = self.isochrone

        ## https://waps.cfa.harvard.edu/MIST/README_tables.pdf
        ## FSPS phase type defined as follows:
        ## -1=PMS, 0=MS, 2=RGB, 3=CHeB, 4=EAGB,
        ##  5=TPAGB, 6=postAGB, 9=WR

        # Main Sequence isochrone integration
        MS = torch.logical_and(isochrone["phase"] >= 0, isochrone["phase"] <= 2)
        # Horizontal Branch isochrone integration
        HB = torch.logical_and(isochrone["phase"] > 2, isochrone["phase"] <= 5)

        #self.ms_weights = self.imf.get_weight(isochrone["initial_mass"][MS], mass_weighted=False)
        #self.hb_weights = self.imf.get_weight(isochrone["initial_mass"][HB], mass_weighted=False)

        imf_weights = self.get_imf_weights(self.isochrone["initial_mass"])
        self.ms_weights = imf_weights[MS]
        self.hb_weights = imf_weights[HB]


        ms_spectra = torch.stack(tuple(
            self.sas.get_spectrum(tf, lg, metalicity,
            ) for lg, tf in zip(isochrone["log_g"][MS], isochrone["Teff"][MS])
        )).T
        ms_spectra *= 10**isochrone["log_l"][MS]

        hb_spectra = torch.stack(tuple(
            self.sas.get_spectrum(tf, lg, metalicity,
            ) for lg, tf in zip(isochrone["log_g"][HB], isochrone["Teff"][HB])
        )).T
        hb_spectra *= 10**isochrone["log_l"][HB]

        spectrum = torch.zeros(ms_spectra.shape[0])

        spectrum += torch.vmap(partial(torch.trapz, x = isochrone["initial_mass"][MS])) (
                        self.ms_weights*ms_spectra
                    )
        spectrum += torch.vmap(partial(torch.trapz, x = isochrone["initial_mass"][HB])) (
                        self.hb_weights*hb_spectra
                    )



        # SSP in L_sun Hz^-1, CvD models in L_sun micron^-1, convert
        # spectrum *= utils.light_speed/self.sas.wavelength**2

        return spectrum


    def plot_isochrone(self, ax):

        #print(len(self.isochrone["log_g"]  ))
        ax.plot(self.isochrone["Teff"], self.isochrone["log_g"], label='Light-House', color='r', lw=3)

    def to(self, dtype=None, device=None):
        self.isochrone.to(dtype=dtype, device=device)
        self.imf.to(dtype=dtype, device=device)
        self.sas.to(dtype=dtype, device=device)

if __name__ == "__main__":
    from isochrone import MIST
    from initial_mass_function import Kroupa
    from stellar_atmosphere_spectrum import PolynomialEvaluator

    ssp = Basic_SSP(MIST(), Kroupa(), PolynomialEvaluator())

    start = time()
    spec = ssp.forward(torch.tensor(0.), torch.tensor(9.), torch.tensor([1.3, 2.3, 2.7]))
    fin = time() - start
    print("runtime: ", fin)

    i = (ssp.sas.wavelength >= 0.36)
    plt.plot(ssp.sas.wavelength[i], spec[i])
    plt.show()
