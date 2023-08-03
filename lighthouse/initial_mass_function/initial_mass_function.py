from abc import ABC, abstractmethod
from torch import Tensor

__all__ = ("Initial_Mass_Function", )

class Initial_Mass_Function(ABC):

    @abstractmethod
    def get_weight(self, mass) -> Tensor:
        pass
