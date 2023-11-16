import torch
from .base import History

__all__ = ("Constant_History", )

class Constant_History(History):

    def __init__(self, value):
        self.value = torch.as_tensor(value)

    def window(self, Tmin, Tmax, **kwargs):
        return self.value
    
    def __call__(self, T, **kwargs):
        return self.value
