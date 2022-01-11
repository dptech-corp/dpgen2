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


class MockPrepDPTrain(PrepDPTrain):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        template = ip['template_script']
        numb_models = ip['numb_models']
        ofiles = []
        osubdirs = []

        for ii in range(numb_models):
            jtmp = template
            jtmp['seed'] = ii
            subdir = Path(f'task.{ii:04d}') 
            subdir.mkdir(exist_ok=True, parents=True)
            fname = subdir / 'input.json'
            with open(fname, 'w') as fp:
                json.dump(jtmp, fp, indent = 4)
            osubdirs.append(str(subdir))
            ofiles.append(fname)

        op = OPIO({
            "task_names" : osubdirs,
            "task_paths" : [Path(ii) for ii in osubdirs],
        })
        return op



