
@ABC
class Initial_Mass_Function():

    @abstractmethod
    def get_weight(self, mass) -> Tensor:
        pass
