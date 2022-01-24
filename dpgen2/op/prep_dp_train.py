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
    r"""Prepares the training scripts and working directories for training
    DP models.

    Inputs of the OP:
    - `template_script`: A template of the training script.
    - `numb_models`: Number of DP models to train.

    Outputs of the OP:
    - `task_names`: The name of tasks. The names of different tasks are different.
    - `task_paths`: The paths to the tasks.
    """

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

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        raise NotImplementedError
    
