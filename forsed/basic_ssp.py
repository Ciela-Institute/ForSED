import torch
from functools import partial
import utils

from time import process_time as time

import matplotlib.pyplot as plt

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

    def forward(self, metalicity, Tage, alpha) -> torch.Tensor:

        isochrone = self.isochrone.get_isochrone(metalicity, Tage)

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
            (spectra * self.imf.get_weight(
                isochrone["initial_mass"][CHOOSE],
                alpha,
            ) * 10**isochrone["log_l"][CHOOSE]),
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
            (spectra * self.imf.get_weight(
                isochrone["initial_mass"][CHOOSE],
                alpha,
            ) * 10**isochrone["log_l"][CHOOSE]),
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

    plt.plot(ssp.sas.wavelength, spec)
    plt.show()
