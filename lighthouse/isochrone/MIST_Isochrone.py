import glob
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
from .read_mist_models import ISO
import h5py
import torch

from .isochrone import Isochrone

__all__ = ("MIST", )

class MIST(Isochrone):

    def __init__(self, iso_file = 'MIST_v1.2_vvcrit0.0_basic_isos.hdf5'):
        data_path      = Path(os.environ['LightHouse_HOME'], 'lighthouse/data/MIST/')


        with h5py.File(os.path.join(data_path, iso_file), 'r') as f:
            self.isochrone_grid = torch.tensor(f["isochrone_grid"][:], dtype = torch.float64)
            self.metallicities = torch.tensor(f["metallicities"][:], dtype = torch.float64)
            self.ages = torch.tensor(f["ages"][:], dtype = torch.float64)
            self.param_order = list(p.decode("UTF-8") for p in f["parameters"][:])

    def get_isochrone(self, metallicity, age, *args, low_m_limit = 0.08, high_m_limit = 100) -> dict:


        metallicity = torch.tensor(metallicity, dtype = torch.float64)
        age         = torch.tensor(age, dtype = torch.float64)

        metallicity_index = torch.isclose(self.metallicities, metallicity, 1e-2).nonzero(as_tuple=False).squeeze()
        age_index = torch.isclose(self.ages, age, 1e-5).nonzero(as_tuple=False).squeeze()


        isochrone = self.isochrone_grid[metallicity_index, age_index].clone() #TODO: do we need to be worried about copy vs deep copy kind of situation here?
        bad_phase_mask = (  (isochrone[3] != 6) & (isochrone[2] >= low_m_limit) & (isochrone[2] <= high_m_limit) )
        isochrone = isochrone[:, bad_phase_mask]

        bad_values = (isochrone[3] > -999)
        isochrone = isochrone[:, bad_values]

        return dict((p, isochrone[i]) for i, p in enumerate(self.param_order))

    def to(self, dtype=None, device=None):
        self.isochrone_grid.to(dtype=dtype, device=device)
        self.metallicities.to(dtype=dtype, device=device)
        self.ages.to(dtype=dtype, device=device)

if __name__=='__main__':
    test = MIST()

    isochrone = test.get_isochrone(0.0, 10.0)

    plt.scatter(isochrone['Teff'], isochrone['log_g'])
    plt.show()
