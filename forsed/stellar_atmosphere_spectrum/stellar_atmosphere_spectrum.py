
@ABC
class Stellar_Atmosphere_Spectrum():
    @abstractmethod
    def get_spectrum(self, logg, Z, Teff) -> Tensor:
        pass

        
