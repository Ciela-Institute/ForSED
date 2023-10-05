from abc import ABC, abstractmethod
from torch import Tensor

__all__ = ("Stellar_Atmosphere_Spectrum", )

class Stellar_Atmosphere_Spectrum(ABC):
    @abstractmethod
    def get_spectrum(self, logg, Z, Teff) -> Tensor:
        ...

    @abstractmethod
    def to(self, dtype=None, device=None):
        ...
        
