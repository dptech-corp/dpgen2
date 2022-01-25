from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
import os, json
from typing import Tuple, List, Set
from pathlib import Path

class RunLmp(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "task_name": str,
            "task_path": Artifact(Path),
            "models" : Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "log" : Artifact(Path),
            "traj" : Artifact(Path),
            "model_devi": Artifact(Path),
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
            - `models`: (`Artifact(List[Path])`) The frozen model to estimate the model deviation. The first model with be used to drive molecular dynamics simulation.

        Returns
        -------
            Output dict with components:
        
            - `traj`: (`Artifact(Path)`) The output trajectory.
            - `model_devi`: (`Artifact(Path)`) The model deviation. The order of recorded model deviations should be consistent with the order of frames in `traj`.
            - `log`: (`Artifact(Path)`) The log file of LAMMPS.
        
        Exceptions
        ----------
        TransientError
            On the failure of LAMMPS execution. Handle different failure cases? e.g. loss atoms.
        """
        raise NotImplementedError


