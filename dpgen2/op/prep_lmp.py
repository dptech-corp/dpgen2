from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)

import json
from typing import Tuple, List
from pathlib import Path
from dpgen2.utils.lmp_task_group import LmpTaskGroup
from dpgen2.constants import (
    lmp_task_pattern,
)

class PrepLmp(OP):
    r"""Prepare configuration files, input scripts for LAMMPS tasks.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "lmp_task_grp": LmpTaskGroup,
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "task_names": List[str],
            "task_paths": Artifact(List[Path]),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip["lmp_task_grp"] : LmpTaskGroup
                Definitions for LAMMPS tasks
        
        Returns
        -------
        op["task_names"]: List[str]
                The name of tasks. Will be used as the identities of the tasks. The names of different tasks are different.
        op["task_paths"]: Artifact(List[Path])
                The parepared working paths of the tasks. The order fo the Paths should be consistent with `op["task_names"]`
        """
        lmp_task_grp = ip['lmp_task_grp']
        cc = 0
        task_paths = []
        for tt in lmp_task_grp:
            ff = tt.files()
            tname = _mk_task_from_files(cc, ff)
            task_paths.append(tname)
            cc += 1
        task_names = [str(ii) for ii in task_paths]
        return OPIO({
            'task_names' : task_names,
            'task_paths' : task_paths,
        })

PrepLmpTaskGroup = PrepLmp

def _mk_task_from_files(cc, ff):
    tname = Path(lmp_task_pattern % cc)
    tname.mkdir(exist_ok=True, parents=True)
    for nn in ff.keys():
        (tname/nn).write_text(ff[nn])
    return tname


        
