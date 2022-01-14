from typing import Dict

class VaspInputs():
    def __init__(
            self,
            incar_template : str,
            potcars : Dict[str, str],
    ):
        self._incar_template = incar_template
        self._potcars = potcars

    @property
    def incar_temp(self):
        return self._incar_template

    @property
    def potcars(self):
        return self._potcars

    def make_potcar(
            self, 
            poscar : str,
    ) -> str:
        raise NotImplementedError

    
