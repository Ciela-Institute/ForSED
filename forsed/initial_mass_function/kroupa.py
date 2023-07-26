
class Kroupa(Initial_Mass_Function):

    def get_weight(self, mass, alpha) -> Tensor:

        weight = torch.where(
            mass < 0.5,
            mass**(-alpha[0]), # mass < 0.5
            torch.where(
                mass < 1.0,
                0.5**(-alpha[0] + alpha[1]) * mass**(-alpha[1]), # 0.5 <= mass < 1.0
                0.5**(-alpha[0] + alpha[1]) * mass**(-alpha[2]), # mass >= 1.0
            )
        )

        return weight
