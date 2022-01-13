from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
import os, json
from typing import Tuple, List, Set
from pathlib import Path

class CollectData(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "name" : str,
            "labeled_data" : Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "labeled_data" : Artifact(Path),
        })
