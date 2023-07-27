
import os
import glob
from pathlib import Path
from time import process_time as time

import pandas as pd 
import torch

from stellar_atmosphere_spectrum import Stellar_Atmosphere_Spectrum

class PolynomialEvaluator(Stellar_Atmosphere_Spectrum):

    def __init__(self):


        directory_path = Path().absolute()
        data_path      = Path(directory_path.parent, 'data/Villaume2017a/')

        self.coefficients = {}
        self.wavelength = {}
        self.reference = {}
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
                self.coefficients[name] = torch.tensor(coeffs.to_numpy()[:,2:], dtype = torch.float64)
                self.wavelength[name] = torch.tensor(coeffs.to_numpy()[:,0], dtype = torch.float64)
                self.reference[name] = torch.tensor(coeffs.to_numpy()[:,1], dtype = torch.float64)

    def get_spectrum(self, surface_gravity, metalicity, effective_temperature) -> torch.Tensor:

        for key, ranges in self.bounds.items():
            if ranges["surface_gravity"][0] < surface_gravity < ranges["surface_gravity"][1] and ranges["effective_temperature"][0] < effective_temperature < ranges["effective_temperature"][1]:
                stellar_type = key
                break

        K = torch.stack((torch.as_tensor(effective_temperature, dtype = torch.float64), torch.as_tensor(metalicity, dtype = torch.float64), torch.as_tensor(10**surface_gravity, dtype = torch.float64)))
        PP = torch.as_tensor(self.polynomial_powers[stellar_type], dtype = torch.float64)
        X = torch.prod(K**PP, dim = -1)
        log_flux = self.coefficients[stellar_type] @ X

        log_flux *= self.reference[stellar_type]

        return self.wavelength[stellar_type], log_flux

if __name__ == "__main__":
    P = PolynomialEvaluator()

    import matplotlib.pyplot as plt
    start = time()
    wave, flux = P.get_spectrum(3., 0., 4500)
    fin = time() - start
    print("runtime: ", fin)
    plt.plot(wave, flux)
    plt.show()
