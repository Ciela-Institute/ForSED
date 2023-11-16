from abc import ABC, abstractmethod

__all__ = ("History", )

class History(ABC):

    @abstractmethod
    def window(self, Tmin, Tmax, **kwargs):
        ...

    @abstractmethod
    def __call__(self, T, **kwargs):
        ...
