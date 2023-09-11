import torch

from scipy.integrate import quad
import numpy as np


from .initial_mass_function import Initial_Mass_Function

__all__ = ("Salpeter", )

class Salpeter(Initial_Mass_Function):

    def get_imf(self, mass, mass_weighted=False) -> torch.Tensor:

        if mass_weighted:
            return mass*mass**(-2.30)
        else:
            return mass**(-2.30) 

    def get_weight(self, mass, ) -> torch.Tensor:
        imf_lower_limit = 0.08
        imf_upper_limit = 100.

        total_mass_weighted_imf, error =  quad(self.get_imf,
                                               imf_lower_limit,
                                               imf_upper_limit,
                                               args=(True,) )

        return self.get_imf(mass)/total_mass_weighted_imf



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    K = Salpeter()

    M = torch.linspace(0.1, 100, 1000)
    IMF = K.get_imf(M)

    plt.plot(torch.log10(M), torch.log10(IMF))
    plt.show()

