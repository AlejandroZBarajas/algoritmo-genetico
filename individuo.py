class Individuo:
    def __init__(
        self,
        id,
        genes=None,
        generacion=0,
        torque=None,
        prob_muta=None
    ):
        self.id = id
        self.genes = genes if genes is not None else []
        self.generacion = generacion
        self.torque = torque
        self.prob_muta = prob_muta

    def __repr__(self):
        return (
            f"Individuo(id= {self.id}, "
            f"gen= {self.generacion}, "
            f"torque= {self.torque},"
            f"prob_muta= {self.prob_muta})"
            
        )
