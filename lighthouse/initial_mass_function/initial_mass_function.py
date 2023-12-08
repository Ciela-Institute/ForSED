from abc import ABC, abstractmethod
# from torch import Tensor

import torch

from scipy.integrate import quad
import numpy as np


__all__ = ("Initial_Mass_Function", )

class Initial_Mass_Function(ABC):

    def __init__(self):

        self.lower_limit = torch.tensor(0.08, dtype=torch.float64)
        self.upper_limit = torch.tensor(100., dtype=torch.float64)


        self._t0_normalization = None


    @abstractmethod
    def get_imf(self, mass, mass_weighted=False) -> torch.Tensor:
        pass


    @property
    def t0_normalization(self): ## TODO: change this name

        if self._t0_normalization is None:
            self._t0_normalization = quad(self.get_imf,
                        self.lower_limit,
                        self.upper_limit,
                        args=(True,) )[0]

        return self._t0_normalization

