import torch




from .initial_mass_function import Initial_Mass_Function

__all__ = ("Salpeter", )

class Salpeter(Initial_Mass_Function):

    def get_imf(self, mass, mass_weighted=False) -> torch.Tensor:

        if mass_weighted:
            return mass*mass**(-2.30)
        else:
            return mass**(-2.30) 



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    K = Salpeter()

    M = torch.linspace(0.08, 100, 1000)
    IMF = K.get_imf(M)

    plt.plot(torch.log10(M), torch.log10(IMF))
    plt.show()

