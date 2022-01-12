from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)

import json
from typing import Tuple, List
from pathlib import Path

class PrepDPTrain(OP):    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "template_script" : dict,
            "numb_models" : int,
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "task_names" : List[str],
            "task_paths" : Artifact(List[Path]),
        })




