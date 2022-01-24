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
    r"""Train and freeze one DP model. 

    A working directory named `task_name` is created. All input files
    are copied or symbol linked to `task_name`, and the DeePMD-kit
    training and freezing tasks are exectuted from `task_name`.

    Inputs of the OP:
    - `task_name`: The name of training task.
    - `task_path`: The path that contains all input files prepareed by `PrepDPTrain`.
    - `init_model`: A frozen model to initialize the training.
    - `init_data`: Initial training data.
    - `iter_data`: Training data generated in the DPGEN iterations.

    Outputs of the OP:
    - `script`: The training script.
    - `model`: The trained frozen model.
    - `lcurve`: The learning curve file.
    - `log`: The log file of training.
    """

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

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        raise NotImplementedError

