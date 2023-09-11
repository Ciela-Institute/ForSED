from abc import ABC, abstractmethod
from torch import Tensor

from scipy.integrate import quad, cumtrapz, simps
import numpy as np


__all__ = ("Initial_Mass_Function", )

class Initial_Mass_Function(ABC):

    @abstractmethod
    def get_imf(self, mass, mass_weighted=False) -> Tensor:
        pass
