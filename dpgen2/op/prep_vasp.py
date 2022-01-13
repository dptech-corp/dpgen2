from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
import os, json
from typing import Tuple, List, Set, Dict
from pathlib import Path

class PrepVasp(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "confs" : Artifact(List[Path]),
            "incar_temp": str,
            "potcars": Dict[str, str],
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "task_names": List[str],
            "task_paths" : Artifact(List[Path]),
        })
