
import os 
import glob
from pathlib import Path



import pandas as pd 

import torch

class PolynomialEvaluator(Stellar_Atmosphere_Spectrum):

    def __init__(self, coefficients):


        directory_path = Path().absolute()
        data_path      = Path(directory_path.parent, 'data/Villaume2017a/')

        self.polynomial_powers = {}
        self.coefficients = {} 

        for file_name in glob.glob(data_path + '*.dat'):

            if 'polynomial_powers' in file_name:

                with open(file_name, 'r') as f:
                    self.polynomial_powers = eval(f.read)

            else: 
                coeffs = pd.read_csv(file_name, delim_whitespace=True, comment='#')
                self.coefficients[os.path.split(file_name)[-1]] = torch.tensor(coeffs)


    def get_spectrum(self, stellar_type, surface_gravity, metalicity, effective_temperature) -> Tensor:


        # Overlap of cool dwarf and warm dwarf training sets
        d_teff_overlap = np.linspace(3000, 5500, num=100)
        d_weights = np.linspace(1, 0, num=100)

        # Overlap of warm giant and hot star training sets
        gh_teff_overlap = np.linspace(5500, 6500, num=100)
        gh_weights = np.linspace(1, 0, num=100)

        # Overlap of warm giant and cool giant training sets
        gc_teff_overlap = np.linspace(3500, 4500, num=100)
        gc_weights = np.linspace(1, 0, num=100)




        # Normalizing to solar values
        logt = np.log10(effective_temperature) - 3.7617
        logg = surface_gravity - 4.44
        feh = metalicity



        X = torch.stack(
            tuple(effective_temperature**c[0] * metalicity**c[1] * surface_gravity**c[2] for p in self.polynomial_powers)
        )
        spectrum = self.coefficients @ X

        return spectrum
