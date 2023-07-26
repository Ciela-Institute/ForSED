


class Basic_SSP(Simulator):

    def __init__(
            self,
            isochrone: "Isochrone",
            imf: "Initial_Mass_Function",
            sas: "Stellar_Atmosphere_Spectrum",
    ):

        self.isochrone = isochrone
        self.imf = imf
        self.sas = sas

    def forward(self, metalicity, Tage, alpha) -> Tensor:

        isochrone = self.isochrone.get_isochrone(metalicity, Tage)

        spectrum = torch.zeros(len)
        
        CHOOSE = isochrone["phase"] <= 2
        spectrum += torch.trapz(
            isochrone["initial_mass"][CHOOSE],
            self.sas.get_spectrum(
                isochrone["logg"][CHOOSE],
                metalicity,
                isochrone["Teff"][CHOOSE]
            ) * self.imf.get_weight(
                isochrone["initial_mass"][CHOOSE],
                alpha
            ) * isochrone["luminosity"][CHOOSE]
        )
            
        CHOOSE = torch.logical_and(isochrone["phase"] > 2, isochrone["phase"] <= 5)
        spectrum += torch.trapz(
            isochrone["initial_mass"][CHOOSE],
            self.sas.get_spectrum(
                isochrone["logg"][CHOOSE],
                metalicity,
                isochrone["Teff"][CHOOSE]
            ) * self.imf.get_weight(
                isochrone["initial_mass"][CHOOSE],
                alpha
            ) * isochrone["luminosity"][CHOOSE]
        )

        # SSP in L_sun Hz^-1, CvD models in L_sun micron^-1, convert
        spectrum *= utils.light_speed/self.sas.wave**2
        
        return spectrum
