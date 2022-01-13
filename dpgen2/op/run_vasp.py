from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
import os, json
from typing import Tuple, List, Set
from pathlib import Path

class RunVasp(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "task_name": str,
            "task_path" : Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "log": Artifact(Path),
            "labeled_data" : Artifact(Path),
        })
