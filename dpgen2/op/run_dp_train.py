from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
import os, json
from typing import Tuple, List, Set
from pathlib import Path

class RunDPTrain(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "task_name" : str,
            "task_path" : Artifact(Path),
            "init_model" : Artifact(Path),
            "init_data" : Artifact(Set[Path]),
            "iter_data" : Artifact(Set[Path]),
        })
    
    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "script" : Artifact(Path),
            "model" : Artifact(Path),
            "lcurve" : Artifact(Path),
            "log" : Artifact(Path),
        })


