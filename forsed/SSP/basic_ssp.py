


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
        
        # Main Sequence isochrone integration
        CHOOSE = isochrone["phase"] <= 2
        spectra = torch.stack(tuple(
            self.sas.get_spectrum(
                lg,
                metalicity,
                tf,
            ) for lg, tf in zip(isochrone["logg"][CHOOSE], isochrone["Teff"][CHOOSE])
        ))
        spectrum = torch.zeros(spectra.shape[1])
        spectrum += torch.trapz(
            spectra * self.imf.get_weight(
                isochrone["initial_mass"][CHOOSE],
                alpha,
            ) * isochrone["luminosity"][CHOOSE],
            isochrone["initial_mass"][CHOOSE],
        )

        # Horizontal Branch isochrone integration
        CHOOSE = torch.logical_and(isochrone["phase"] > 2, isochrone["phase"] <= 5)
        spectra = torch.stack(tuple(
            self.sas.get_spectrum(
                lg,
                metalicity,
                tf,
            ) for lg, tf in zip(isochrone["logg"][CHOOSE], isochrone["Teff"][CHOOSE])
        ))
        spectrum += torch.trapz(
            spectra * self.imf.get_weight(
                isochrone["initial_mass"][CHOOSE],
                alpha,
            ) * isochrone["luminosity"][CHOOSE],
            isochrone["initial_mass"][CHOOSE],
        )

        # SSP in L_sun Hz^-1, CvD models in L_sun micron^-1, convert
        spectrum *= utils.light_speed/self.sas.wavelength**2
        
        return spectrum
