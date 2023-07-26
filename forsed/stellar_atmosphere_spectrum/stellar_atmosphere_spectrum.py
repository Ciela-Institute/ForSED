from abc import ABC, abstractmethod
from torch import Tensor

class Stellar_Atmosphere_Spectrum(ABC):
    @abstractmethod
    def get_spectrum(self, logg, Z, Teff) -> Tensor:
        pass

        
