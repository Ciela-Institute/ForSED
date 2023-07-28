import os 
from pathlib import Path

def get_polynomial_coefficients_villaume2017a():
    """
    
    """

    import requests

    base_name = 'https://raw.githubusercontent.com/AlexaVillaume/SPI_Utils/master/spigen/Coefficients/{0}.dat'

    stellar_types = ['Cool_Dwarfs', 'Cool_Giants', 'Warm_Dwarfs', 'Warm_Giants', 'Hot_Stars']


    directory_path = Path().absolute()
    data_path      = Path(directory_path.parent, 'data/Villaume2017a/')

    try:
        os.mkdir(data_path)
    except Exception as e:
        pass

    for regime in stellar_types:

        # check if file already exists
        file_path = os.path.join(data_path, regime+'.dat')

        if os.path.exists(file_path):
            continue


        r = requests.get(base_name.format(regime)) 
        with open(file_path, 'w') as f:
            f.write(r.text) 

    C = {}
    C['Cool_Dwarfs'] = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [0,2,0], [2,0,0], [0,0,2], [1,1,0], [1,0,1], [0,1,1], [0,3,0], [3,0,0], [0,0,3], [2,1,0], [1,2,0], [2,0,1], [4,0,0], [0,4,0], [2,2,0], [3,1,0], [5,0,0]]
    C['Cool_Giants'] = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [2,0,0], [0,0,2], [0,2,0], [0,1,1], [1,0,1], [1,1,0], [3,0,0], [0,0,3], [0,3,0], [1,1,1], [2,1,0], [2,0,1], [1,2,0], [0,2,1], [1,0,2], [0,1,2], [4,0,0]]
    C['Warm_Dwarfs'] = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [2,0,0], [0,0,2], [0,2,0], [1,1,0], [1,0,1], [3,0,0], [1,0,2], [0,3,0], [2,0,1], [2,1,0], [1,2,0], [1,1,1], [0,2,1], [4,0,0], [0,0,4], [3,0,1], [3,1,0], [2,2,0], [1,3,0], [2,0,2], [2,1,1], [5,0,0]]
    C['Warm_Giants'] = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [2,0,0], [0,0,2], [0,2,0], [1,1,0], [1,0,1], [0,1,1], [3,0,0], [0,0,3], [0,3,0], [2,1,0], [1,2,0], [2,0,1], [1,0,2], [4,0,0], [0,4,0], [2,2,0], [2,0,2], [0,2,2], [5,0,0]]
    C['Hot_Stars']   = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [2,0,0], [0,2,0], [0,0,2], [1,0,1], [1,1,0], [0,1,1], [3,0,0], [0,0,3], [0,3,0], [1,1,1], [2,1,0], [2,0,1], [1,2,0], [0,2,1], [1,0,2], [0,1,2], [4,0,0]]

    with open(os.path.join(data_path, 'polynomial_powers.dat'), "w") as f: 
        f.write(str(C))


    B = {
        "Cool_Dwarfs": {"surface_gravity": (-0.5, 3.5), "effective_temperature": (2500,4000)},
        "Cool_Giants": {"surface_gravity": (3.5, 6),    "effective_temperature": (2500,4000)},
        "Warm_Dwarfs": {"surface_gravity": (-0.5, 3.5), "effective_temperature": (4000,6000)},
        "Warm_Giants": {"surface_gravity": (3.5, 6),    "effective_temperature": (4000,6000)},
        "Hot_Stars": {"surface_gravity":   (-0.5, 6),   "effective_temperature": (6000,12000)},
    }

    with open(os.path.join(data_path, 'bounds.dat'), "w") as f: 
        f.write(str(B))
    

if __name__=='__main__': 
    get_polynomial_coefficients_villaume2017a()
