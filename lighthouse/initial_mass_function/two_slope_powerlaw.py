import torch


from .initial_mass_function import Initial_Mass_Function

__all__ = ("Two_Slope_Powerlaw", )

class Two_Slope_Powerlaw(Initial_Mass_Function):

    """
    Fixed lower mass cutoff
    Fixed break 
    """

    def get_imf(self, mass, mass_weighted=False, alpha1=1.0, alpah2=1.0) -> torch.Tensor:

        alpha = torch.tensor([alpha1, alpha2, -2.30])
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

    
if __name__ == "__main__":

    pass