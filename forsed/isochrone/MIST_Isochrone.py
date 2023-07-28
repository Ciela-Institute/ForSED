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

    def __init__(self, iso_file = 'MIST_v1.2_vvcrit0.4_basic_isos.hdf5'):
        directory_path = Path(__file__).parent
        data_path      = Path(directory_path.parent, 'data/MIST/')

        with h5py.File(os.path.join(data_path, iso_file), 'r') as f:
            self.isochrone_grid = torch.tensor(f["isochrone_grid"][:], dtype = torch.float64)
            self.metallicities = torch.tensor(f["metallicities"][:], dtype = torch.float64)
            self.ages = torch.tensor(f["ages"][:], dtype = torch.float64)
            self.param_order = list(p.decode("UTF-8") for p in f["parameters"][:])

    def get_isochrone(self, metallicity, age, *args, low_m_limit = 0.08, high_m_limit = 100) -> dict:

        metallicity_index = torch.clamp(torch.sum(self.metallicities < metallicity) - 1, 0) 
        age_index = torch.clamp(torch.sum(self.ages < age) - 1, 0) # TODO: figure out a better way later

        isochrone = self.isochrone_grid[metallicity_index, age_index]
        isochrone = isochrone[:,isochrone[3] > -999]

        return dict((p, isochrone[i]) for i, p in enumerate(self.param_order))

if __name__=='__main__':
    test = MIST()

    isochrone = test.get_isochrone(0.0, 10.0)

    plt.scatter(isochrone['Teff'], isochrone['log_g'])
    plt.show()
