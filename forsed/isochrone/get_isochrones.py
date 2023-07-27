"""
Methods to download isochrone libraries.
"""


import os
from pathlib import Path
import tarfile
from time import sleep


def get_mist_isochrones(iso_version = 'MIST_v1.2_vvcrit0.4_basic_isos.txz', url='https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/{}'):

    import requests

    directory_path = Path().absolute()
    data_path      = Path(directory_path.parent, 'data/MIST/')
    
    try:
        os.mkdir(data_path)
    except FileExistsError as e:
        pass

    # check if file already exists
    file_path = os.path.join(data_path, iso_version)

    if not os.path.exists(os.path.splitext(file_path)[0]):
        print("Getting MIST")
        r = requests.get(url.format(iso_version))
        print("Writing MIST")
        with open(file_path, 'wb') as f:
            f.write(r.content)
        sleep(2)
        print("Extracting MIST")
        T = tarfile.open(file_path)
        T.extractall(data_path)
        print("Deleting tar file")
        os.remove(file_path)
        

if __name__=='__main__':

    get_mist_isochrones()
