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
    r"""Prepares the working directories for VASP tasks.

    A list of (same length as ip["confs"]) working directories
    containing all files needed to start VASP tasks will be
    created. The paths of the directories will be returned as
    `op["task_paths"]`. The identities of the tasks are returned as
    `op["task_names"]`.

    """

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

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:

            - `inputs` : (`VaspInputs`) Definitions for the VASP inputs
            - `confs` : (`Artifact(List[Path])`) Configuration files for the VASP tasks. 
        
        Returns
        -------
        op : dict 
            Output dict with components:

            - `task_names`: (`List[str]`) The name of tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `task_paths`: (`Artifact(List[Path])`) The parepared working paths of the tasks. Contains all input files needed to start the VASP. The order fo the Paths should be consistent with `op["task_names"]`
        """
        raise NotImplementedError
