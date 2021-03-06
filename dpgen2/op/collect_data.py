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
    """Collect labeled data and add to the iteration dataset.

    After running FP tasks, the labeled data are scattered in task
    directories.  This OP collect the labeled data in one data
    directory and add it to the iteration data. The data generated by
    this iteration will be place in `ip["name"]` subdirectory of the
    iteration data directory.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "name" : str,
            "labeled_data" : Artifact(List[Path]),
            "iter_data" : Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "iter_data" : Artifact(List[Path]),
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
        
            - `name`: (`str`) The name of this iteration. The data generated by this iteration will be place in a sub-directory of `name`.
            - `labeled_data`: (`Artifact(List[Path])`) The paths of labeled data generated by FP tasks.
            - `iter_data`: (`Artifact(set[Path])`) The paths of iteration data.

        Returns
        -------
            Output dict with components:
        
            - `iter_data`: (`Artifact(List[Path])`) The paths of iteration data, added with labeled data generated by this iteration.
        
        """
        raise NotImplementedError

