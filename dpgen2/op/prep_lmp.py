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

class PrepLmp(OP):
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


def mk_task_from_files(cc, ff):
    tname = Path(f'task.{cc:06d}')
    tname.mkdir(exist_ok=True, parents=True)
    for nn in ff.keys():
        (tname/nn).write_text(ff[nn])
    return tname


class PrepLmpTaskGroup(PrepLmp):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        lmp_task_grp = ip['lmp_task_grp']
        cc = 0
        task_paths = []
        for tt in lmp_task_grp:
            ff = tt.files()
            tname = mk_task_from_files(cc, ff)
            task_paths.append(tname)
            cc += 1
        task_names = [str(ii) for ii in task_paths]
        return OPIO({
            'task_names' : task_names,
            'task_paths' : task_paths,
        })
        
