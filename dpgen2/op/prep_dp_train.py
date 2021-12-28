from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
from typing import Tuple, List
from pathlib import Path

class PrepDPTrain(OP):    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "template_script" : dict,
            "numb_models" : int,
            "init_data" : Artifact(Set[Path]),
            "iter_data" : Artifact(Set[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "train_scripts" : Artifact(List[Path]),
        })


class MockPrepDPTrain(PrepDPTrain):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        template = ip['template_script']
        numb_models = ip['numb_models']
        init_data = ip['init_data']
        iter_data = ip['iter_data']
        ofiles = []

        for ii in range(numb_models):
            jtmp = template
            jtmp['seed'] = ii
            fname = Path(f'task.{ii:4d}') / 'input.json'
            with open(fname, 'w') as fp:
                json.dump(jtmp, fp, indent = 4)

        op = OPIO({
            "train_scripts" : ofiles,
        })
        return op
