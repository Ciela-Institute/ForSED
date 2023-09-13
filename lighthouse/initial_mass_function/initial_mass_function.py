from abc import ABC, abstractmethod
# from torch import Tensor

import torch 

from scipy.integrate import quad
import numpy as np


__all__ = ("Initial_Mass_Function", )

class Initial_Mass_Function(ABC):

    def __init__(self, ):

        self._normalization = None

        self.lower_limit = 0.08
        self.upper_limit = 100.


    @abstractmethod
    def get_imf(self, mass, mass_weighted=False) -> torch.Tensor:
        pass

    @property 
    def normalization(self): 

        if self._normalization is None:
            self._normalization = quad(self.get_imf,
                        self.lower_limit,
                        self.upper_limit,
                        args=(True,) )[0]
            
        return self._normalization
    

    def get_weight(self, mass) -> torch.Tensor:

        # Normalizing to 1 solar mass at t=0
        # (or that's the goal at least)
        return self.get_imf(mass, mass_weighted=False)/self.normalization
    

    def get_stellar_mass(self,  isochrone) -> torch.Tensor:
        pop_masses = isochrone["initial_mass"]

        weights = []
        for i, mass in enumerate(pop_masses):
            if pop_masses[i] < self.lower_limit or pop_masses[i] > self.upper_limit:
                print("Bounds of isochrone exceed limits of full IMF")
            if i == 0:
                m1 = pop_masses[i]
            else:
                m1 = pop_masses[i] - 0.5*(pop_masses[i] - pop_masses[i-1])
            if i == len(pop_masses) - 1:
                m2 = pop_masses[i]
            else:
                m2 = pop_masses[i] + 0.5*(pop_masses[i+1] - pop_masses[i])
            
            if m2 < m1:
                print("IMF_WEIGHT WARNING: non-monotonic mass!", m1, m2, m2-m1)
                continue 
            
            if m2 == m1:
                print("m2 == m1")
                continue

            tmp, error =  quad(self.get_imf, 
                                m1, 
                                m2, 
                                args=(False,) )
            weights.append(tmp)

        weights = np.asarray(weights)

        imf_weight = weights/self.normalization
    
        print(sum(imf_weight))

        stellar_mass = (torch.sum(isochrone["current_mass"]*imf_weight))
        formed_mass  = (torch.sum(isochrone["initial_mass"]*imf_weight))

        print(stellar_mass, formed_mass)