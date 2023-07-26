class Isochrone(object):

    @abstractmethod
    def get_isochrone(self, metalicity, Tage, *args, low_m_limit = 0.08, high_m_limit = 100) -> dict:
        pass #phase, stellar_mass, Teff, logg, logL