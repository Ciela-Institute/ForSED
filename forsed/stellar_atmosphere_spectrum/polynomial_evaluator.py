

class PolynomialEvaluator(Stellar_Atmosphere_Spectrum):

    def __init__(self, coefficients):

        pass

    def get_spectrum(self, surface_gravity, metalicity, effective_temperature) -> Tensor:

        spectrum = self.coefficients @ (...)

        return spectrum
