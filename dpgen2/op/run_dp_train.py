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


class MockRunDPTrain(RunDPTrain):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        work_dir = Path(ip['task_name'])
        script = ip['task_path'] / 'input.json'
        init_model = Path(ip['init_model'])
        init_data = ip['init_data']
        iter_data = ip['iter_data']

        script = Path(script).resolve()
        init_model = init_model.resolve()
        init_model_str = str(init_model)
        init_data = [ii.resolve() for ii in init_data]
        iter_data = [ii.resolve() for ii in iter_data]
        init_data_str = [str(ii) for ii in init_data]
        iter_data_str = [str(ii) for ii in iter_data]

        with open(script) as fp:
            jtmp = json.load(fp)        
        data = []
        for ii in sorted(init_data_str):
            data.append(ii)
        for ii in sorted(iter_data_str):
            data.append(ii)
        jtmp['data'] = data
        with open(script, 'w') as fp:
            json.dump(jtmp, fp, indent=4)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        oscript = Path('input.json')
        if not oscript.exists():
            from shutil import copyfile
            copyfile(script, oscript)
        model = Path('model.pb')
        lcurve = Path('lcurve.out')
        log = Path('log')

        assert(init_model.exists())        
        with log.open("w") as f:
            f.write(f'init_model {str(init_model)} OK\n')
        for ii in jtmp['data']:
            assert(Path(ii).exists())
            assert((ii in init_data_str) or (ii in iter_data_str))
            with log.open("a") as f:
                f.write(f'data {str(ii)} OK\n')
        assert(script.exists())
        with log.open("a") as f:
            f.write(f'script {str(script)} OK\n')

        with model.open("w") as f:
            f.write('read from init model: \n')
            f.write(init_model.read_text() + '\n')
        with lcurve.open("w") as f:
            f.write('read from train_script: \n')
            f.write(script.read_text() + '\n')

        os.chdir(cwd)
        
        return OPIO({
            'script' : work_dir/oscript,
            'model' : work_dir/model,
            'lcurve' : work_dir/lcurve,
            'log' : work_dir/log
        })


