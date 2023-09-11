import torch

from scipy.integrate import quad
import numpy as np


from .initial_mass_function import Initial_Mass_Function

__all__ = ("Kroupa", )

class Kroupa(Initial_Mass_Function):

    def get_imf(self, mass, mass_weighted=False) -> torch.Tensor:

        alpha = torch.tensor([1.3, 2.3, 2.3])
        mass  = torch.tensor(mass)

        imf = torch.where(
            mass < 0.5,
            mass**(-alpha[0]), # mass < 0.5
            torch.where(
                mass < 1.0,
                0.5**(-alpha[0] + alpha[1]) * mass**(-alpha[1]), # 0.5 <= mass < 1.0
                0.5**(-alpha[0] + alpha[1]) * mass**(-alpha[2]), # mass >= 1.0
            )
        )

        if mass_weighted:
            return mass*imf
        else:
            return imf
        
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
    K = Kroupa()

    M = torch.linspace(0.1, 100, 1000)
    IMF = K.get_imf(M)

    plt.plot(torch.log10(M), torch.log10(IMF))
    plt.show()
    
