


class Basic_SSP(Simulator):

    def __init__(self, isochrone, imf, f_get_sasy):

        self.isochrone = isochrone
        self.imf = imf
        self.f_get_sasy = f_get_sasy

    def forward(self, metalicity, Tage, alpha) -> Tensor:

        isochrone = self.isochrone.get_isochrone(metalicity, Tage)

        spectrum = torch.zeros(len)
        for i in range(len(isochrone["phase"])):
            elif isochrone["phase"][i] <= 2:
                normalized_spectrum = self.f_get_sasy.get_spectrum(metalicity, Tage)
                weight = self.imf.get_weight(isochrone["initial_mass"][i], alpha)
                weight *= isochrone["initial_mass"][i] - isochrone["initial_mass"][i-1]
                spectrum += normalized_spectrum * isochrone["luminosity"][i] * weight
            elif isochrone["phase"][i] <= 5:
                pass
            else:
                continue
