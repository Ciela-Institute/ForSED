

class PolynomialEvaluator(Stellar_Atmosphere_Spectrum):

    def __init__(self, coefficients):

        pass

    def get_spectrum(self, stellar_type, surface_gravity, metalicity, effective_temperature) -> Tensor:

        K = torch.stack((effective_temperature, metalicity, surface_gravity))
        if stellar_type == "cool dwarf":
            C = [[0,0,0], [1,0,0], [0,1,0], [0,0,1],[0,2,0],[2,0,0],[0,0,2], [1,1,0], [1,0,1], [0,1,1],[0,3,0],[3,0,0],[0,0,3],[2,1,0],[1,2,0],[2,0,1],[4,0,0],[0,4,0],[0,0,4],[2,2,0],[3,1,0], [5,0,0]]
        elif stellar_type == "cool giants":
            C = [[0,0,0], [1,0,0], [0,1,0], [0,0,1],[2,0,0],[0,0,2],[0,2,0], [0,1,1], [1,0,1], [1,1,0],[3,0,0],[0,0,3],[0,3,0],[1,1,1],[2,1,0],[2,0,1],[1,2,0],[0,2,1],[1,0,2],[0,1,2],[4,0,0]]
        elif stellar_type == "warm dwarfs":
            C = [[0,0,0], [1,0,0], [0,1,0], [0,0,1],[2,0,0],[0,0,2],[0,2,0], [1,1,0], [1,0,1],[3,0,0],[1,0,2],[0,3,0],[2,1,0],[2,0,1],[1,0,2],[1,1,1],[0,2,1],[4,0,0],[0,4,0],[3,1,0],[3,0,1],[2,2,0],[1,3,0],[2,0,2],[2,1,1],[5,0,0]]
        elif stellar_type == "warm giants":
            C = [[0,0,0], [1,0,0], [0,1,0], [0,0,1],[2,0,0],[0,0,2],[0,2,0], [1,1,0], [1,0,1], [0,1,1], [3,0,0], [0,0,3], [0,3,0], [2,1,0], [1,2,0], [2,0,1], [1,0,2], [4,0,0], [0,4,0],[2,2,0],[2,0,2],[0,2,2], [5,0,0]]
        elif stellar_type == "hot stars":
            C = [[0,0,0], [1,0,0], [0,1,0], [0,0,1],[2,0,0], [0,2,0],[0,0,2],[1,0,1], [1,1,0], [0,1,1], [3,0,0], [0,0,3],[0,3,0], [1,1,1], [2,1,0], [2,0,1], [0,2,1], [1,0,2], [0,1,2], [4,0,0]]
        else:
            raise ValueError(f"unknown stellar type: {stellar_type}")
        X = torch.stack(
            tuple(K[0]**c[0] * K[1]**c[1] * K[2]**c[2] for c in C)
        )
        spectrum = self.coefficients @ X

        return spectrum
