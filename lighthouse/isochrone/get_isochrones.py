"""
Methods to download isochrone libraries.
"""


import os
from pathlib import Path
import tarfile
from time import sleep
import shutil
from glob import glob

import h5py
import numpy as np
from .read_mist_models import ISO

__all__ = ("get_mist_isochrones", )


def get_mist_isochrones(iso_version = 'MIST_v1.2_vvcrit0.0_basic_isos.txz', url='https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/{}'):


    # Collect isochrone data from the internet
    ######################################################################
    import requests

    # Path to where MIST data will live
    directory_path = Path().absolute()
    data_path      = Path(directory_path, 'lighthouse/data/MIST/')

    # Ensure the directoty exists to place the files
    try:
        os.mkdir(data_path)
    except FileExistsError as e:
        pass

    # Specific file path for the requested version of MIST
    file_path = os.path.join(data_path, iso_version)

    # Skip download if files already exit
    if not os.path.exists(os.path.splitext(file_path)[0]):
        # Pull the isochrone files from the internet
        print("Getting MIST")
        r = requests.get(url.format(iso_version))

        # Write the tar file to disk
        print("Writing MIST")
        with open(file_path, 'wb') as f:
            f.write(r.content)

        # Extract the tar file into the individual .iso files
        print("Extracting MIST")
        T = tarfile.open(file_path)
        T.extractall(data_path)

        # Remove the old tar file, no longer needed
        print("Deleting tar file")
        os.remove(file_path)

    # Collect isochrones into data structure
    ######################################################################
    print("Compiling MIST data")
    # Get the isochrone files
    isochrone_files = glob(os.path.join(os.path.splitext(file_path)[0], '*.iso'))

    # Run through all the files and collect information about metalicities and ages
    metallicities = []
    longest_track = 0
    for isochrone_file in isochrone_files:
        isochrone = ISO(isochrone_file, verbose=False)

        metallicities.append(isochrone.abun['[Fe/H]'])
        for age in isochrone.ages:
            i = isochrone.age_index(age)
            tracklength = len(isochrone.isos[i]['log_g'])
            if longest_track < tracklength:
                longest_track = tracklength

    isochrone_grid = np.zeros((len(isochrone_files), isochrone.num_ages, 6, longest_track)) - 999


    # Go through the isochrone files and collect all the data
    for n, isochrone_file in enumerate(isochrone_files):


        isochrone = ISO(isochrone_file, verbose=False)
        for x, age in enumerate(isochrone.ages):

            i = isochrone.age_index(age)

            track_length = len(isochrone.isos[i]['log_g'])

            isochrone_grid[n, x, 0][:track_length] = np.array(isochrone.isos[i]['log_g'])
            isochrone_grid[n, x, 1][:track_length] = np.array(10**isochrone.isos[i]['log_Teff'])
            isochrone_grid[n, x, 2][:track_length] = np.array(isochrone.isos[i]['initial_mass'])
            isochrone_grid[n, x, 3][:track_length] = np.array(isochrone.isos[i]['phase'])
            isochrone_grid[n, x, 4][:track_length] = np.array(isochrone.isos[i]['log_L'])
            isochrone_grid[n, x, 5][:track_length] = np.array(isochrone.isos[i]['star_mass'])


    ######################################################################
    print("Writing MIST to hdf5 database")
    with h5py.File(os.path.splitext(file_path)[0] + ".hdf5", 'w') as f:
        # Isochrne grid
        data_iso_grid = f.create_dataset("isochrone_grid", data = isochrone_grid)
        data_iso_grid.attrs["description"] = "This is a 4D tensor of isochrones orgnaized by: metalicity, age, parameter, track length. Parameter has length 6 and goes by: log_g, Teff, initial_mass, phase, log_L, star_mass"

        # Meta data for each axis
        data_metallicities = f.create_dataset("metallicities", data = metallicities)
        data_metallicities.attrs["description"] = "For the metallicity axis of the isochrone_grid, this is the associated matalicities"

        data_ages = f.create_dataset("ages", data = isochrone.ages)
        data_ages.attrs["description"] = "For the ages axis of the isochrone_grid, this is the associated ages"

        dt = h5py.special_dtype(vlen=str)
        params = ["log_g", "Teff", "initial_mass", "phase", "log_l", "current_mass"]
        data_params = f.create_dataset("parameters", (len(params), ), dtype = dt)
        data_params[:] = params
        data_params.attrs["description"] = "For the parameters axis of the isochrone_grid, this lists the relevant parameters in the correct order"

    ## Cleanup
    #shutil.rmtree(os.path.splitext(file_path)[0]) # TODO: this should be optional, with the default keeping the .iso files

if __name__=='__main__':

    get_mist_isochrones()
