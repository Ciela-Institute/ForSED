import glob
from re import M 

import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
from .read_mist_models import ISO

from .isochrone import Isochrone

import torch 

__all__ = ("MIST", )

class MIST(Isochrone):

    def __init__(self):
        directory_path = Path(__file__).parent
        data_path      = Path(directory_path.parent, 'data/MIST/')
        
        isochrone_files = glob.glob(os.path.join(data_path, 'MIST_v1.2_vvcrit0.4_basic_isos/*.iso'))
        isochrone_grid = torch.zeros(len(isochrone_files), 107, 5, 1700) - 999

        metallicities = []
        for isochrone_file in isochrone_files: 

            isochrone = ISO(isochrone_file, verbose=False)
            metallicities.append(isochrone.abun['[Fe/H]'])

            ages = [round(x, 2) for x in isochrone.ages]
        

        self.metallicities = torch.tensor(list(sorted(metallicities)))
        self.ages          = torch.tensor(ages)

        metallicities_order = np.argsort(metallicities)

        for n, isochrone_file in enumerate(isochrone_files): 

            isochrone = ISO(isochrone_file, verbose=False)

            loggs = []
            teffs = []
            log_Ls = []
            initial_masses = [] 
            phases = []
            for x, age in enumerate(isochrone.ages):
                i = isochrone.age_index(age)

                j = np.where((isochrone.isos[i]['phase'] != 6) &
                    (isochrone.isos[i]['initial_mass'] >= 0.08) &
                    (isochrone.isos[i]['initial_mass'] <= 100.)
                )

                something = len(isochrone.isos[i]['log_g'][j])
                isochrone_grid[metallicities_order[n], i, 0][:something] = torch.tensor(isochrone.isos[i]['log_g'][j])
                isochrone_grid[metallicities_order[n], i, 1][:something] = torch.tensor(10**isochrone.isos[i]['log_Teff'][j])
                isochrone_grid[metallicities_order[n], i, 2][:something] = torch.tensor(isochrone.isos[i]['initial_mass'][j])
                isochrone_grid[metallicities_order[n], i, 3][:something] = torch.tensor(isochrone.isos[i]['phase'][j])
                isochrone_grid[metallicities_order[n], i, 4][:something] = torch.tensor(isochrone.isos[i]['log_L'][j])
                
        self.isochrone_grid = isochrone_grid

    def get_isochrone(self, metallicity, age, *args, low_m_limit = 0.08, high_m_limit = 100) -> dict:

        metallicity_index = torch.clamp(torch.sum(self.metallicities < metallicity) - 1, 0) 
        age_index = torch.clamp(torch.sum(self.ages < age) - 1, 0) # TO-DO: figure out a better way later


        isochrone = self.isochrone_grid[metallicity_index, age_index]

        isochrone = isochrone[:,isochrone[3] > -999]

        return {'log_g': isochrone[0], 
                'Teff': isochrone[1], 
                'initial_mass': isochrone[2],
                'phase': isochrone[3], 
                'log_l': isochrone[4]
        }

if __name__=='__main__':
    test = MIST()

    isochrone = test.get_isochrone(0.0, 10.0)

    plt.scatter(isochrone['Teff'], isochrone['log_g'])
    plt.show()
