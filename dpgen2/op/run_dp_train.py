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
    r"""Execute a DP training task. Train and freeze a DP model. 

    A working directory named `task_name` is created. All input files
    are copied or symbol linked to directory `task_name`. The
    DeePMD-kit training and freezing commands are exectuted from
    directory `task_name`.

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
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:
        
            - `task_name`: (`str`) The name of training task.
            - `task_path`: (`Artifact(Path)`) The path that contains all input files prepareed by `PrepDPTrain`.
            - `init_model`: (`Artifact(Path)`) A frozen model to initialize the training.
            - `init_data`: (`Artifact(set[Path])`) Initial training data.
            - `iter_data`: (`Artifact(set[Path])`) Training data generated in the DPGEN iterations.

        Returns
        -------
            Output dict with components:
        
            - `script`: (`Artifact(Path)`) The training script.
            - `model`: (`Artifact(Path)`) The trained frozen model.
            - `lcurve`: (`Artifact(Path)`) The learning curve file.
            - `log`: (`Artifact(Path)`) The log file of training.
        
        Exceptions
        ----------
        FatalError
            On the failure of training or freezing. Human intervention needed.
        """
        raise NotImplementedError

