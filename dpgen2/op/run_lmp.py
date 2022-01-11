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



class MockedRunLmp(RunLmp):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        task_name = ip['task_name']
        task_path = ip['task_path']
        models = ip['models']
        
        task_path = task_path.resolve()
        models = [ii.resolve() for ii in models]
        models_str = [str(ii) for ii in models]
        
        work_dir = Path(task_name)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        import glob
        ifiles = glob.glob(str(task_path / '*'))
        for ii in ifiles:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)
        for ii in models:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)
        
        log = Path('log')
        traj = Path('dump.traj')
        model_devi = Path('model_devi.out')
        
        # fc = ['log of {task_name}']
        # for ii in ['conf.lmp', 'in.lammps'] + models_str:
        #     if Path(ii).exists():
        #         fc.append(f'{ii} OK')
        # log.write_text('\n'.join(fc))        
        # log.write_text('log of {task_name}')
        fc = []
        for ii in ['conf.lmp', 'in.lammps'] + [ii.name for ii in models]:
             fc.append(Path(ii).read_text())
        log.write_text('\n'.join(fc))
        traj.write_text(f'traj of {task_name}')
        model_devi.write_text(f'model_devi of {task_name}')

        os.chdir(cwd)

        return OPIO({
            'log' : work_dir/log,
            'traj' : work_dir/traj,
            'model_devi' : work_dir/model_devi,
        })
        
