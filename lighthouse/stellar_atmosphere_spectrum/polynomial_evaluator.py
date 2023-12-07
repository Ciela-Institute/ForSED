
import os
import glob
from pathlib import Path
from time import process_time as time

import numpy as np

import pandas as pd
import torch

from .stellar_atmosphere_spectrum import Stellar_Atmosphere_Spectrum

__all__ = ("PolynomialEvaluator", )

class PolynomialEvaluator(Stellar_Atmosphere_Spectrum):

    def __init__(self):

        data_path      = Path(os.environ['LightHouse_HOME'], 'lighthouse/data/Villaume2017a/')


        self.coefficients = {}
        self.reference    = {}
        for file_name in glob.glob(str(data_path) + '/*.dat'):
            if 'polynomial_powers' in file_name:
                with open(file_name, 'r') as f:
                    self.polynomial_powers = eval(f.read())
                    for key in self.polynomial_powers:
                        self.polynomial_powers[key] = torch.tensor(self.polynomial_powers[key])
            elif "bounds" in file_name:
                with open(file_name, 'r') as f:
                    self.bounds = eval(f.read())
            else:
                coeffs = pd.read_csv(file_name, delim_whitespace=True, comment='#')
                name = os.path.split(file_name)[-1][:-4]

                self.wavelength         = torch.tensor(coeffs.to_numpy()[:,0], dtype  = torch.float64)
                self.reference[name]    = torch.tensor(coeffs.to_numpy()[:,1], dtype  = torch.float64)
                self.coefficients[name] = torch.tensor(coeffs.to_numpy()[:,2:], dtype = torch.float64)

    def get_spectrum(self, teff, logg, feh) -> torch.Tensor:

        """
        These weights are used later to ensure
        smooth behavior.
        """
        # Overlap of cool dwarf and warm dwarf training sets
        d_teff_overlap = torch.linspace(3000, 5500, steps=100)
        d_weights = torch.linspace(1, 0, steps=100)

        # Overlap of warm giant and hot star training sets
        gh_teff_overlap = torch.linspace(5500, 6500, steps=100)
        gh_weights = torch.linspace(1, 0, steps=100)

        # Overlap of warm giant and cool giant training sets
        gc_teff_overlap = torch.linspace(3500, 4500, steps=100)
        gc_weights = torch.linspace(1, 0, steps=100)



        """
        Setting up some boundaries
        """
        teff2 = teff
        logg2 = logg
        if teff2 <= 2800.:
            teff2 = torch.tensor(2800)
        if logg2 < (-0.5):
            logg2 = torch.tensor(-0.5)

        # Normalizing to solar values
        logt = np.log10(teff2) - 3.7617
        #logt = np.log10(teff) - 3.7617
        logg = logg - 4.44

        for key, ranges in self.bounds.items():
            if ranges["surface_gravity"][0] <= logg2 <= ranges["surface_gravity"][1] and ranges["effective_temperature"][0] <= teff2 <= ranges["effective_temperature"][1]:

                if key is 'Hot_Giants' or key is 'Hot_Dwarfs':
                    stellar_type = 'Hot_Stars'
                else:
                    stellar_type = key
                break


        K  = torch.stack((torch.as_tensor(logt, dtype = torch.float64),
                          torch.as_tensor(feh,  dtype = torch.float64),
                          torch.as_tensor(logg, dtype = torch.float64)))

        easy_types = ['Cool_Giants', 'Warm_Giants', 'Cool_Dwarfs', 'Warm_Dwarfs', 'Hot_Stars']

        if stellar_type in easy_types:

            PP = torch.as_tensor(self.polynomial_powers[stellar_type], dtype = torch.float64)
            X  = torch.prod(K**PP, dim = -1)

            flux = np.exp(self.coefficients[stellar_type] @ X)
            flux *= self.reference[stellar_type]

        elif stellar_type is 'Hottish_Giants':

            PP = torch.as_tensor(self.polynomial_powers['Warm_Giants'], dtype = torch.float64)
            X  = torch.prod(K**PP, dim = -1)

            flux1 = np.exp(self.coefficients['Warm_Giants'] @ X)
            flux1 *= self.reference['Warm_Giants']

            PP = torch.as_tensor(self.polynomial_powers['Hot_Stars'], dtype = torch.float64)
            X  = torch.prod(K**PP, dim = -1)

            flux2 = np.exp(self.coefficients['Hot_Stars'] @ X)
            flux2 *= self.reference['Hot_Stars']

            t_index = (np.abs(gh_teff_overlap - teff2)).argmin()
            weight = gh_weights[t_index]
            flux = (flux1*weight + flux2*(1-weight))

        elif stellar_type is 'Coolish_Giants':

            PP = torch.as_tensor(self.polynomial_powers['Warm_Giants'], dtype = torch.float64)
            X  = torch.prod(K**PP, dim = -1)

            flux1 = np.exp(self.coefficients['Warm_Giants'] @ X)
            flux1 *= self.reference['Warm_Giants']

            PP = torch.as_tensor(self.polynomial_powers['Cool_Giants'], dtype = torch.float64)
            X  = torch.prod(K**PP, dim = -1)

            flux2 = np.exp(self.coefficients['Cool_Giants'] @ X)
            flux2 *= self.reference['Cool_Giants']

            t_index = (np.abs(gh_teff_overlap - teff2)).argmin()
            weight = gc_weights[t_index]
            flux = (flux1*weight + flux2*(1-weight))

        elif stellar_type is 'Coolish_Dwarfs':

            PP = torch.as_tensor(self.polynomial_powers['Warm_Dwarfs'], dtype = torch.float64)
            X  = torch.prod(K**PP, dim = -1)

            flux1 = np.exp(self.coefficients['Warm_Dwarfs'] @ X)
            flux1 *= self.reference['Warm_Dwarfs']

            PP = torch.as_tensor(self.polynomial_powers['Cool_Dwarfs'], dtype = torch.float64)
            X  = torch.prod(K**PP, dim = -1)

            flux2 = np.exp(self.coefficients['Cool_Dwarfs'] @ X)
            flux2 *= self.reference['Warm_Dwarfs']

            t_index = (np.abs(gh_teff_overlap - teff2)).argmin()
            weight = d_weights[t_index]
            flux = (flux1*weight + flux2*(1-weight))

        else:
            error = ('Parameter out of bounds:'
                     'teff = {0},  logg {1}')
            raise ValueError(error.format(teff2, logg))


        return flux

    def to(self, dtype=None, device=None):
        self.wavelength.to(dtype=dtype, device=device)
        for key in self.polynomial_powers:
            self.polynomial_powers[key].to(dtype=dtype, device=device)
        self.bounds.to(dtype=dtype, device=device)
        for name in self.reference:
            self.reference[name].to(dtype=dtype, device=device)
            self.coefficients[name].to(dtype=dtype, device=device)



if __name__ == "__main__":

    pass
