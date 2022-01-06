from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
from typing import Tuple, List
from pathlib import Path

class RunDPTrain(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "task_subdir": str,
            "train_script" : Artifact(Path),
            "init_model" : Artifact(Path),
            "init_data" : Artifact(Set[Path]),
            "iter_data" : Artifact(Set[Path]),
        })
    
    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "model" : Artifact(Path),
            "lcurve" : Artifact(Path),
            "log" : Artifact(Path),
        })


class MockRunDPTrain(OP):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        script = ip['train_script']
        init_model = ip['init_model']
        init_data = ip['init_data']
        iter_data = ip['iter_data']
        work_dir = Path(ip['task_subdir'])

        with open(script) as fp:
            jtmp = json.load(fp)        
        data = []
        for ii in init_data:
            data.append(ii)
        for ii in iter_data:
            data.append(ii)
        jtmp['data'] = data
        with open(script) as fp:
            json.dump(jtmp, fp, indent=4)

        cwd = os.getcwd()
        work_dir.mkdir()
        os.chdir(work_dir)
        
        model = Path('model.pb')
        lcurve = Path('lcurve.out')
        log = Path('log')

        assert(init_model.exists())
        log.write_text(f'init model {str(init_model)} OK')
        for ii in script['data']:
            assert(ii.exists())
            assert((ii in init_data) or (ii in iter_data))
            log.write_text('data {str(ii)} OK')
        assert(script.exists())
        log.write_text('script {str(script)} OK')

        model.write('read from init model: ')
        model.write(init_model.read_text())
        lcurve.write('read from train_script')
        lcurve.write(script.read_text())        

        os.chdir(cwd)
        
        return OPIO({
            'model' : work_dir/model,
            'lcurve' : work_dir/lcurve,
            'log' : work_dir/log
        })
