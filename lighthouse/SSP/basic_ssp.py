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
            isochrone: "Isochrone",
            imf: "Initial_Mass_Function",
            sas: "Stellar_Atmosphere_Spectrum",
    ):

        self.isochrone_grid = isochrone
        self.imf = imf
        self.sas = sas

        self.ms_weights = None
        self.hb_weights = None

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



    @property
    def tage_normalization(self):
        # Normalizing to 1 solar mass at each epoch,

        if self._tage_normalization is None:
            self._tage_normalization = quad(self.imf.get_imf,
                        self.imf.lower_limit,
                        self.highest_mass,
                        args=(True,) )[0]

        return self._tage_normalization

    def get_weight(self, mass, mass_weighted=True) -> torch.Tensor:
        return self.imf.get_imf(mass, mass_weighted=mass_weighted)/self.tage_normalization

    def get_stellar_mass(self,  pop_masses) -> torch.Tensor:
        # pop_masses = isochrone["initial_mass"]

        weights = []
        for i, mass in enumerate(pop_masses):
            if pop_masses[i] < self.lower_limit or pop_masses[i] > self.upper_limit:
                print("Bounds of isochrone exceed limits of full IMF")
            if i == 0:
                m1 = pop_masses[i]
            else:
                m1 = pop_masses[i] - 0.5*(pop_masses[i] - pop_masses[i-1])
            if i == len(pop_masses) - 1:
                m2 = pop_masses[i]
            else:
                m2 = pop_masses[i] + 0.5*(pop_masses[i+1] - pop_masses[i])

            if m2 < m1:
                print("IMF_WEIGHT WARNING: non-monotonic mass!", m1, m2, m2-m1)
                continue

            if m2 == m1:
                print("m2 == m1")
                continue

            tmp, error =  quad(self.get_imf,
                                m1,
                                m2,
                                args=(False,) )
            weights.append(tmp)

        weights = np.asarray(weights)

        imf_weight = weights/self.t0_normalization

        return imf_weight



    def forward(self, metalicity, Tage, alpha) -> torch.Tensor:

        isochrone = self.isochrone_grid.get_isochrone(metalicity, Tage)
        self.highest_mass = isochrone["initial_mass"].max()

        ## https://waps.cfa.harvard.edu/MIST/README_tables.pdf
        ## FSPS phase type defined as follows:
        ## -1=PMS, 0=MS, 2=RGB, 3=CHeB, 4=EAGB,
        ##  5=TPAGB, 6=postAGB, 9=WR

        # Main Sequence isochrone integration
        MS = torch.logical_and(isochrone["phase"] >= 0, isochrone["phase"] <= 2)
        # Horizontal Branch isochrone integration
        HB = torch.logical_and(isochrone["phase"] > 2, isochrone["phase"] <= 5)

        self.ms_weights = self.imf.get_weight(isochrone["initial_mass"][MS], mass_weighted=False)
        self.hb_weights = self.imf.get_weight(isochrone["initial_mass"][HB], mass_weighted=False)

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

        #print(torch.sum(self.ms_weights * isochrone["current_mass"][MS]) + torch.sum(self.hb_weights * isochrone["current_mass"][HB]) )
        #print(torch.sum(self.ms_weights * isochrone["initial_mass"][MS]) + torch.sum(self.hb_weights * isochrone["initial_mass"][HB]) )

        summed_weights = self.ms_weights.sum().item() + self.hb_weights.sum().item()
        #print(summed_weights)


        # SSP in L_sun Hz^-1, CvD models in L_sun micron^-1, convert
        # spectrum *= utils.light_speed/self.sas.wavelength**2

        return spectrum


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
