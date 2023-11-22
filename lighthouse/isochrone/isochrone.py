from abc import ABC, abstractmethod

__all__ = ("Isochrone", )

class Isochrone(ABC):

    @abstractmethod
    def get_isochrone(self, metalicity, Tage, *args, low_m_limit = 0.08, high_m_limit = 100) -> dict:
        ... #phase, stellar_mass, Teff, logg, logL

    @abstractmethod
    def to(self, dtype=None, device=None):
        ...
