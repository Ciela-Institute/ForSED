"""
Methods to download isochrone libraries.
"""


import os
from pathlib import Path


def get_mist_isochrones(url='https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/{}'):

    import requests
        
    iso_version = 'MIST_v1.2_vvcrit0.4_basic_isos.txz'


    directory_path = Path().absolute()
    data_path      = Path(directory_path.parent, 'data/MIST/')
    
    try:
        os.mkdir(data_path)
    except Exception as e:
        print(e)
        pass

    # check if file already exists
    file_path = os.path.join(data_path, iso_version)

    if not os.path.exists(file_path):
        

        r = requests.get(url.format(iso_version))

        with open(file_path, 'w') as f:
            f.write(r.text)

    ## Still need to figure out how to extrat the .txz files


if __name__=='__main__':

    get_mist_isochrones()