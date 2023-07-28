
from time import process_time as time

import numpy as np

import spigen



from stellar_atmosphere_spectrum import PolynomialEvaluator


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    teff = 3110.1831261806437
    logg = 5.1249530724684771
    feh  = 0.0

    P = PolynomialEvaluator()
    sas = P.get_spectrum(logg, feh, teff)



    spec = spigen.Spectrum()
    spec = spec.from_coefficients(teff, logg, feh)


    i = (P.wavelength >= 0.36)
    plt.plot(P.wavelength[i], sas[i])
    plt.plot(spec['wave'], spec['flux'], color='k')

    plt.show()