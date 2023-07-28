
from time import process_time as time

import numpy as np

import spigen



from stellar_atmosphere_spectrum import PolynomialEvaluator


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    teff = 4110.1831261806437
    logg = 4.1249530724684771
    feh  = 0.0

    P = PolynomialEvaluator()
    sas = P.get_spectrum(teff, logg, feh)

    spec = spigen.Spectrum()
    spec = spec.from_coefficients(teff, logg, feh)

    i = (P.wavelength >= 0.36)
    plt.plot(spec['wave'], spec['flux'], color='k', lw=3, label='SPI_Utils')
    plt.plot(P.wavelength[i], sas[i], label='polynomial_evaluator')

    plt.legend()

    plt.show()