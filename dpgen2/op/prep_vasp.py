from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
import os, json
from typing import Tuple, List, Set, Dict
from pathlib import Path
from dpgen2.fp.vasp import VaspInputs

class PrepVasp(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "inputs": VaspInputs,
            "confs" : Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "task_names": List[str],
            "task_paths" : Artifact(List[Path]),
        })
