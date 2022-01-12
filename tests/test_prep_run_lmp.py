import os
import numpy as np
import unittest

from dflow import (
    InputParameter,
    OutputParameter,
    Inputs,
    InputArtifact,
    Outputs,
    OutputArtifact,
    Workflow,
    Step,
    Steps,
    upload_artifact,
    download_artifact,
    S3Artifact,
    argo_range
)
from dflow.python import (
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    upload_packages,
)

import time, shutil, json, jsonpickle
from typing import Set, List
from pathlib import Path

try:
    from context import dpgen2
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from dpgen2.op.prep_lmp import PrepLmpTaskGroup
from dpgen2.op.run_lmp import RunLmp
from dpgen2.flow.prep_run_lmp import prep_run_lmp
from dpgen2.op.lmp_task_group import LmpTask, LmpTaskGroup

upload_packages.append(__file__)

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
        


def make_task_group_list(ngrp, ntask_per_grp):
    tgrp_list = []
    for ii in range(ngrp):
        tgrp = LmpTaskGroup()
        for jj in range(ntask_per_grp):
            tt = LmpTask()
            tt\
                .add_file('conf.lmp', f'group{ii} task{jj} conf')\
                .add_file('in.lammps', f'group{ii} task{jj} input')
            tgrp.add_task(tt)
        tgrp_list.append(tgrp)
    return tgrp_list


def check_lmp_tasks(tcase, ngrp, ntask_per_grp):
    cc = 0
    tdirs = []
    for ii in range(ngrp):
        for jj in range(ntask_per_grp):
            tdir = f'task.{cc:06d}'
            tdirs.append(tdir)
            tcase.assertTrue(Path(tdir).is_dir())
            fconf = Path(tdir)/'conf.lmp'
            finpt = Path(tdir)/'in.lammps'
            tcase.assertTrue(fconf.is_file())
            tcase.assertTrue(finpt.is_file())
            tcase.assertEqual(fconf.read_text(), f'group{ii} task{jj} conf')
            tcase.assertEqual(finpt.read_text(), f'group{ii} task{jj} input')
            cc += 1
    return tdirs


class TestPrepLmpTaskGroup(unittest.TestCase):
    def setUp(self):
        self.ngrp = 2
        self.ntask_per_grp = 3
        self.task_group_list = make_task_group_list(self.ngrp, self.ntask_per_grp)
        
    def tearDown(self):
        for ii in range(self.ngrp * self.ntask_per_grp):
            work_path = Path(f'task.{ii:06d}')
            if work_path.is_dir():
                shutil.rmtree(work_path)

    def test(self):
        op = PrepLmpTaskGroup()
        out = op.execute( OPIO({
            'lmp_task_grps' : self.task_group_list,
        }) )
        tdirs = check_lmp_tasks(self, self.ngrp, self.ntask_per_grp)
        tdirs = [str(ii) for ii in tdirs]

        self.assertEqual(tdirs, out['task_names'])
        self.assertEqual(tdirs, [str(ii) for ii in out['task_paths']])



class TestMockedRunLmp(unittest.TestCase):
    def setUp(self):
        self.ntask = 2
        self.nmodels = 3
        self.task_list = []
        self.model_list = []
        for ii in range(self.ntask):
            work_path = Path(f'task.{ii:06d}')
            work_path.mkdir(exist_ok=True, parents=True)
            (work_path/'conf.lmp').write_text(f'conf {ii}')
            (work_path/'in.lammps').write_text(f'input {ii}')
            self.task_list.append(work_path)
        for ii in range(self.nmodels):
            model = Path(f'model{ii}.pb')
            model.write_text(f'model {ii}')
            self.model_list.append(model)

    def check_run_lmp_output(
            self,
            task_name : str,
            models : List[Path],
    ):
        cwd = os.getcwd()
        os.chdir(task_name)
        fc = []
        for ii in ['conf.lmp', 'in.lammps'] + [ii.name for ii in models]:
            fc.append(Path(ii).read_text())    
        self.assertEqual(fc, Path('log').read_text().strip().split('\n'))
        self.assertEqual(f'traj of {task_name}', Path('dump.traj').read_text())
        self.assertEqual(f'model_devi of {task_name}', Path('model_devi.out').read_text())
        os.chdir(cwd)


    def tearDown(self):
        for ii in range(self.ntask):
            work_path = Path(f'task.{ii:06d}')
            if work_path.is_dir():
                shutil.rmtree(work_path)
        for ii in range(self.nmodels):
            model = Path(f'model{ii}.pb')
            if model.is_file():
                os.remove(model)
            
    def test(self):
        self.task_list_str = [str(ii) for ii in self.task_list]
        self.model_list_str = [str(ii) for ii in self.model_list]
        for ii in range(self.ntask):
            ip = OPIO({
                'task_name' : self.task_list_str[ii],
                'task_path' : self.task_list[ii],
                'models' : self.model_list,
            })
            op = MockedRunLmp()
            out = op.execute(ip)
            self.assertEqual(out['log'] , Path(f'task.{ii:06d}')/'log')
            self.assertEqual(out['traj'] , Path(f'task.{ii:06d}')/'dump.traj')
            self.assertEqual(out['model_devi'] , Path(f'task.{ii:06d}')/'model_devi.out')
            self.assertTrue(out['log'].is_file())
            self.assertTrue(out['traj'].is_file())
            self.assertTrue(out['model_devi'].is_file())
            self.check_run_lmp_output(self.task_list_str[ii], self.model_list)


class TestPrepRunLmp(unittest.TestCase):
    def setUp(self):
        self.ngrp = 2
        self.ntask_per_grp = 3
        self.task_group_list = make_task_group_list(self.ngrp, self.ntask_per_grp)
        self.nmodels = 3
        self.model_list = []
        for ii in range(self.nmodels):
            model = Path(f'model{ii}.pb')
            model.write_text(f'model {ii}')
            self.model_list.append(model)
        self.models = upload_artifact(self.model_list)

    def tearDown(self):
        for ii in range(self.nmodels):
            model = Path(f'model{ii}.pb')
            if model.is_file():
                os.remove(model)
        for ii in range(self.ngrp * self.ntask_per_grp):
            work_path = Path(f'task.{ii:06d}')
            if work_path.is_dir():
                shutil.rmtree(work_path)
        

    def check_run_lmp_output(
            self,
            task_name : str,
            models : List[Path],
    ):
        cwd = os.getcwd()
        os.chdir(task_name)
        fc = []
        idx = int(task_name.split('.')[1])
        ii = idx // self.ntask_per_grp
        jj = idx - ii * self.ntask_per_grp
        fc.append(f'group{ii} task{jj} conf')
        fc.append(f'group{ii} task{jj} input')
        for ii in [ii.name for ii in models]:
            fc.append((Path('..')/Path(ii)).read_text())    
        self.assertEqual(fc, Path('log').read_text().strip().split('\n'))
        self.assertEqual(f'traj of {task_name}', Path('dump.traj').read_text())
        self.assertEqual(f'model_devi of {task_name}', Path('model_devi.out').read_text())
        os.chdir(cwd)


    def test(self):
        steps = prep_run_lmp(
            "prep-run-lmp",
            PrepLmpTaskGroup,
            MockedRunLmp,
        )        
        prep_run_step = Step(
            'prep-run-step', 
            template = steps,
            parameters = {
                "lmp_task_grps" : self.task_group_list,
            },
            artifacts = {
                "models" : self.models,
            },
        )

        wf = Workflow(name="dp-train")
        wf.add(prep_run_step)
        wf.submit()
        
        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(4)

        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step(name="prep-run-step")[0]
        self.assertEqual(step.phase, "Succeeded")

        download_artifact(step.outputs.artifacts["model_devis"])
        download_artifact(step.outputs.artifacts["trajs"])
        download_artifact(step.outputs.artifacts["logs"])

        for ii in jsonpickle.decode(step.outputs.parameters['task_names'].value):
            self.check_run_lmp_output(ii, self.model_list)
            
