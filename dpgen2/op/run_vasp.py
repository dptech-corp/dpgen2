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
    r"""Execute a VASP task.

    A working directory named `task_name` is created. All input files
    are copied or symbol linked to directory `task_name`. The VASP
    command is exectuted from directory `task_name`. The
    `op["labeled_data"]` in `"deepmd/npy"` format (HF5 in the future)
    provided by `dpdata` will be created.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "task_name": str,
            "task_path" : Artifact(Path),
            "config" : dict,
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "log": Artifact(Path),
            "labeled_data" : Artifact(Path),
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
        
            - `task_name`: (`str`) The name of task.
            - `task_path`: (`Artifact(Path)`) The path that contains all input files prepareed by `PrepVasp`.

        Returns
        -------
            Output dict with components:
        
            - `log`: (`Artifact(Path)`) The log file of VASP.
            - `labeled_data`: (`Artifact(Path)`) The path to the labeled data in `"deepmd/npy"` format provided by `dpdata`.
        
        Exceptions
        ----------
        TransientError
            On the failure of VASP execution. 
        """
        raise NotImplementedError

