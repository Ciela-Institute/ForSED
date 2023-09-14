from abc import ABC, abstractmethod
# from torch import Tensor

import torch 

from scipy.integrate import quad
import numpy as np


__all__ = ("Initial_Mass_Function", )

class Initial_Mass_Function(ABC):

    def __init__(self):

        self._t0_normalization = None

        self.lower_limit = 0.08
        self.upper_limit = 100.


    @abstractmethod
    def get_imf(self, mass, mass_weighted=False) -> torch.Tensor:
        pass

    @property 
    def t0_normalization(self): 

        # Normalizing to 1 solar mass at t=0
        # (or that's the goal at least)

        if self._t0_normalization is None:
            self._t0_normalization = quad(self.get_imf,
                        self.lower_limit,
                        self.upper_limit,
                        args=(True,) )[0]
            
        return self._t0_normalization
    

    def get_weight(self, mass, mass_weighted=False) -> torch.Tensor:
        return self.get_imf(mass, mass_weighted=mass_weighted)/self.t0_normalization
