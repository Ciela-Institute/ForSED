
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

        directory_path = Path(__file__).parent
        data_path      = Path(directory_path.parent, 'data/Villaume2017a/')

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

                self.wavelength   = torch.tensor(coeffs.to_numpy()[:,0], dtype = torch.float64)
                self.reference[name]    = torch.tensor(coeffs.to_numpy()[:,1], dtype = torch.float64)
                self.coefficients[name] = torch.tensor(coeffs.to_numpy()[:,2:], dtype = torch.float64)


    def get_spectrum(self, surface_gravity, metalicity, effective_temperature) -> torch.Tensor:

        for key, ranges in self.bounds.items():
            if ranges["surface_gravity"][0] < surface_gravity < ranges["surface_gravity"][1] and ranges["effective_temperature"][0] < effective_temperature < ranges["effective_temperature"][1]:
                stellar_type = key
                break
        else:
            stellar_type = "Warm_Giants"

        K = torch.stack((torch.as_tensor(effective_temperature, dtype = torch.float64), torch.as_tensor(metalicity, dtype = torch.float64), torch.as_tensor(surface_gravity, dtype = torch.float64)))
        PP = torch.as_tensor(self.polynomial_powers[stellar_type], dtype = torch.float64)
        X = torch.prod(K**PP, dim = -1)
        log_flux = self.coefficients[stellar_type] @ X

        log_flux *= self.reference[stellar_type]

        return log_flux

    # def get_spectrum(self, logg, feh, teff) -> torch.Tensor:

    #     print(logg, feh, teff)


    #     """
    #     Setting up some boundaries
    #     """
    #     teff2 = teff
    #     logg2 = logg
    #     if teff2 <= 2800.:
    #         teff2 = 2800
    #     if logg2 < (-0.5):
    #         logg2 = (-0.5)

    #     # Normalizing to solar values
    #     logt = np.log10(teff2) - 3.7617
    #     logg = logg - 4.44

    #     #for key, ranges in self.bounds.items():
    #         # print(key, ranges)

    #     stellar_type = 'Hot_Stars'
            

    #     K = torch.stack((torch.as_tensor(teff, dtype = torch.float64), torch.as_tensor(feh, dtype = torch.float64), torch.as_tensor(logg, dtype = torch.float64)))
    #     PP = torch.as_tensor(self.polynomial_powers[stellar_type], dtype = torch.float64)
    #     X = torch.prod(K**PP, dim = -1)

    #     log_flux = self.coefficients[stellar_type] @ X
    #     log_flux *= self.reference[stellar_type]

    #     return self.wavelength[stellar_type], log_flux



if __name__ == "__main__":
    P = PolynomialEvaluator()

    # import read_mist_models as read_mist
    import matplotlib.pyplot as plt

    # isochrone = read_mist.ISO('../data/MIST/MIST_v1.2_vvcrit0.4_basic_isos/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_basic.iso')

    # age = 10.0

    # i = isochrone.age_index(age)
    # j = ((isochrone.isos[i]['phase'] != 3) &
    #     (isochrone.isos[i]['phase'] != 4) &
    #     (isochrone.isos[i]['phase'] != 5) &
    #     (isochrone.isos[i]['phase'] != 6))

    # teff = isochrone.isos[i]['log_Teff'][j]
    # logg = isochrone.isos[i]['log_g'][j]

    feh = 0.0
    teff = [7000]
    logg = [4.0]

    for i, (t, g) in enumerate(zip(teff, logg)):
        fig = plt.figure(figsize=(12,6))
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
        ax1.scatter(teff, logg, color='#999999', s=40)
        ax1.set_xlim(3.8, 3.4)
        ax1.set_ylim(5.5, -0.5)

        wave, flux = P.get_spectrum(g, feh, t)
        # basis = spigen.Spectrum()
        # spec = basis.from_coefficients(10**t, g, feh)
        ax1.scatter(t, g, color='#ef8a62', s=70)
        ax2.plot(wave, flux, color='#ef8a62')

        ax1.set_xlabel('Temperature', fontsize=24)
        ax1.set_ylabel('Surface Gravity', fontsize=24)

        ax2.set_xlabel('Wavelength', fontsize=24)
        ax2.set_ylabel('Flux', fontsize=24)

        plt.show()



    start = time()
    wave, flux = P.get_spectrum(3., 0., 4500)
    fin = time() - start
    print("runtime: ", fin)
    plt.plot(wave, flux)
    plt.show()
