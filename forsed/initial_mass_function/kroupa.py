import torch

from .initial_mass_function import Initial_Mass_Function

__all__ = ("Kroupa", )

class Kroupa(Initial_Mass_Function):

    def get_weight(self, mass, alpha) -> torch.Tensor:

        weight = torch.where(
            mass < 0.5,
            mass**(-alpha[0]), # mass < 0.5
            torch.where(
                mass < 1.0,
                0.5**(-alpha[0] + alpha[1]) * mass**(-alpha[1]), # 0.5 <= mass < 1.0
                0.5**(-alpha[0] + alpha[1]) * mass**(-alpha[2]), # mass >= 1.0
            )
        )

        return weight

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    K = Kroupa()

    M = torch.linspace(0.1, 100, 1000)
    W = K.get_weight(M, torch.tensor([1.3, 2.3, 2.7]))

    plt.plot(torch.log10(M), torch.log10(W))
    plt.show()
    
