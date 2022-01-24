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
    r"""Prepares the training scripts for DP training tasks.
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
        r"""Execute the OP.

        Parameters
        ----------
        ip["template_script"]: dict
                A template of the training script.
        ip["numb_models"]: int
                Number of DP models to train.
        
        Returns
        -------
        op["task_names"]: List[str]
                The name of tasks. Will be used as the identities of the tasks. The names of different tasks are different.
        op["task_paths"]: Artifact(List[Path])
                The parepared working paths of the tasks. The order fo the Paths should be consistent with `op["task_names"]`
        """
        raise NotImplementedError
    
