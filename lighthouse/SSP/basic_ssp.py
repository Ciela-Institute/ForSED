import torch
from functools import partial
from time import process_time as time

import matplotlib.pyplot as plt

from scipy.integrate import quad
import numpy as np

from .. import utils

class Basic_SSP():

    def __init__(
            self,
            isochrone: "Isochrone",
            imf: "Initial_Mass_Function",
            sas: "Stellar_Atmosphere_Spectrum",
    ):

        self.isochrone = isochrone
        self.imf = imf
        self.sas = sas

        self.spectrum     = None
        self.formed_mass  = None
        self.stellar_mass = None

    # def get_weight(self) -> torch.Tensor:

    #     return None

    def get_stellar_mass(self,  isochrone) -> torch.Tensor:

        # !compute IMF-weighted mass of the SSP
        # mass_ssp(ii) = SUM(wght(1:nmass(i))*mact(i,1:nmass(i)))


        pop_masses = isochrone["initial_mass"]
        imf_lower_limit = 0.08
        imf_upper_limit = 100.
            
        weights = []
        for i, mass in enumerate(pop_masses):
            if pop_masses[i] < imf_lower_limit or pop_masses[i] > imf_upper_limit:
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
                # print "IMF_WEIGHT WARNING: non-monotonic mass!", m1, m2, m2-m1
                continue 
            
            if m2 == m1:
                # print("m2 == m1")
                continue

            tmp, error =  quad(self.imf.get_imf, 
                                m1, 
                                m2, 
                                args=(False,) )
            weights.append(tmp)

        weights = np.asarray(weights)

        total_imf, error =  quad(self.imf.get_imf, 
                                        imf_lower_limit, 
                                        imf_upper_limit, 
                                        args=(False,) )

        imf_weight = weights/total_imf
    

        self.stellar_mass = (torch.sum(isochrone["current_mass"]*imf_weight))

    def forward(self, metalicity, Tage, alpha) -> torch.Tensor:

        isochrone = self.isochrone.get_isochrone(metalicity, Tage)
        self.get_stellar_mass(isochrone)



        # Main Sequence isochrone integration
        CHOOSE = isochrone["phase"] <= 2

        spectra = torch.stack(tuple(
            self.sas.get_spectrum(
               tf,
               lg,
                metalicity,
            ) for lg, tf in zip(isochrone["log_g"][CHOOSE], isochrone["Teff"][CHOOSE])
        )).T

        spectrum = torch.zeros(spectra.shape[0])

        spectrum += torch.vmap(partial(torch.trapz, x = isochrone["initial_mass"][CHOOSE]))(
            ( (spectra * 10**isochrone["log_l"][CHOOSE]) * self.imf.get_weight(
                isochrone["initial_mass"][CHOOSE]
            ) ),
        )



        # Horizontal Branch isochrone integration
        CHOOSE = torch.logical_and(isochrone["phase"] > 2, isochrone["phase"] <= 5)

        spectra = torch.stack(tuple(
            self.sas.get_spectrum(
                tf,
                lg,
                metalicity,
            ) for lg, tf in zip(isochrone["log_g"][CHOOSE], isochrone["Teff"][CHOOSE])
        )).T

        spectrum += torch.vmap(partial(torch.trapz, x = isochrone["initial_mass"][CHOOSE]))(
            ( (spectra * 10**isochrone["log_l"][CHOOSE]) * self.imf.get_weight(
                isochrone["initial_mass"][CHOOSE]
            ) ),
        )



        # SSP in L_sun Hz^-1, CvD models in L_sun micron^-1, convert
        spectrum *= utils.light_speed/self.sas.wavelength**2

        return spectrum
    



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
