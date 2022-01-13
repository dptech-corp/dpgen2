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
from dpgen2.op.prep_vasp import PrepVasp
from dpgen2.op.run_vasp import RunVasp
from dpgen2.flow.prep_run_fp import prep_run_fp

upload_packages.append(__file__)

class MockedPrepVasp(PrepVasp):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        confs = ip['confs']
        incar_temp = ip['incar_temp']
        potcars = ip['potcars']
        
        nconfs = len(confs)
        task_paths = []

        for ii in range(nconfs):
            task_path = Path(f'task.{ii:06d}')
            task_path.mkdir(exist_ok=True, parents=True)
            from shutil import copyfile
            copyfile(confs[ii], task_path/'POSCAR')
            (task_path/'INCAR').write_text(incar_temp)
            task_paths.append(task_path)

        task_names = [str(ii) for ii in task_paths]
        return OPIO({
            'task_names' : task_names,
            'task_paths' : task_paths,
        })


class MockedRunVasp(RunVasp):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        task_name = ip['task_name']
        task_path = ip['task_path']

        work_dir = Path(task_name)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        import glob
        ifiles = glob.glob(str(task_path / '*'))
        for ii in ifiles:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)
        
        log = Path('log')
        labeled_data = Path('labeled_data')
        
        fc = []
        for ii in ['POSCAR', 'INCAR']:
             fc.append(Path(ii).read_text())
        log.write_text('\n'.join(fc))
        labeled_data.write_text(f'labeled_data of {task_name}')

        os.chdir(cwd)

        return OPIO({
            'log' : work_dir/log,
            'labeled_data' : work_dir/labeled_data,
        })
        



def check_vasp_tasks(tcase, ntasks):
    cc = 0
    tdirs = []
    for ii in range(ntasks):
        tdir = f'task.{cc:06d}'
        tdirs.append(tdir)
        tcase.assertTrue(Path(tdir).is_dir())
        fconf = Path(tdir)/'POSCAR'
        finpt = Path(tdir)/'INCAR'
        tcase.assertTrue(fconf.is_file())
        tcase.assertTrue(finpt.is_file())
        tcase.assertEqual(fconf.read_text(), f'conf {ii}')
        tcase.assertEqual(finpt.read_text(), f'incar template')
        cc += 1
    return tdirs


class TestPrepVaspTaskGroup(unittest.TestCase):
    def setUp(self):
        self.ntasks = 6
        self.confs = []
        for ii in range(self.ntasks):
            fname = Path(f'conf.{ii}')
            fname.write_text(f'conf {ii}')
            self.confs.append(fname)
        self.incar = 'incar template'
        
    def tearDown(self):
        for ii in range(self.ntasks):
            work_path = Path(f'task.{ii:06d}')
            if work_path.is_dir():
                shutil.rmtree(work_path)
            fname = Path(f'conf.{ii}')
            os.remove(fname)

    def test(self):
        op = MockedPrepVasp()
        out = op.execute( OPIO({
            'confs' : self.confs,
            'incar_temp' : self.incar,
            'potcars' : {'foo': 'bar'},
        }) )
        tdirs = check_vasp_tasks(self, self.ntasks)
        tdirs = [str(ii) for ii in tdirs]
        self.assertEqual(tdirs, out['task_names'])
        self.assertEqual(tdirs, [str(ii) for ii in out['task_paths']])


class TestMockedRunVasp(unittest.TestCase):
    def setUp(self):
        self.ntask = 6
        self.task_list = []
        for ii in range(self.ntask):
            work_path = Path(f'task.{ii:06d}')
            work_path.mkdir(exist_ok=True, parents=True)
            (work_path/'POSCAR').write_text(f'conf {ii}')
            (work_path/'INCAR').write_text(f'incar template')
            self.task_list.append(work_path)

    def check_run_lmp_output(
            self,
            task_name : str,
    ):
        cwd = os.getcwd()
        os.chdir(task_name)
        fc = []
        for ii in ['POSCAR', 'INCAR']:
            fc.append(Path(ii).read_text())    
        self.assertEqual(fc, Path('log').read_text().strip().split('\n'))
        self.assertEqual(f'labeled_data of {task_name}', Path('labeled_data').read_text())
        os.chdir(cwd)

    def tearDown(self):
        for ii in range(self.ntask):
            work_path = Path(f'task.{ii:06d}')
            if work_path.is_dir():
                shutil.rmtree(work_path)
            
    def test(self):
        self.task_list_str = [str(ii) for ii in self.task_list]
        for ii in range(self.ntask):
            ip = OPIO({
                'task_name' : self.task_list_str[ii],
                'task_path' : self.task_list[ii],
            })
            op = MockedRunVasp()
            out = op.execute(ip)
            self.assertEqual(out['log'] , Path(f'task.{ii:06d}')/'log')
            self.assertEqual(out['labeled_data'] , Path(f'task.{ii:06d}')/'labeled_data')
            self.assertTrue(out['log'].is_file())
            self.assertTrue(out['labeled_data'].is_file())
            self.check_run_lmp_output(self.task_list_str[ii])


class TestPrepRunVasp(unittest.TestCase):
    def setUp(self):
        self.ntasks = 6
        self.confs = []
        for ii in range(self.ntasks):
            fname = Path(f'conf.{ii}')
            fname.write_text(f'conf {ii}')
            self.confs.append(fname)
        self.incar = 'incar template'
        self.confs = upload_artifact(self.confs)

    def tearDown(self):
        for ii in range(self.ntasks):
            work_path = Path(f'task.{ii:06d}')
            if work_path.is_dir():
                shutil.rmtree(work_path)
            fname = Path(f'conf.{ii}')
            os.remove(fname)        

    def check_run_vasp_output(
            self,
            task_name : str,
    ):
        cwd = os.getcwd()
        os.chdir(task_name)
        fc = []
        ii = int(task_name.split('.')[1])
        fc.append(f'conf {ii}')
        fc.append(f'incar template')
        self.assertEqual(fc, Path('log').read_text().strip().split('\n'))
        self.assertEqual(f'labeled_data of {task_name}', Path('labeled_data').read_text())
        os.chdir(cwd)

    def test(self):
        steps = prep_run_fp(
            "prep-run-vasp",
            MockedPrepVasp,
            MockedRunVasp,
        )        
        prep_run_step = Step(
            'prep-run-step', 
            template = steps,
            parameters = {
                "incar_temp" : self.incar,
                "potcars" : {'foo' : 'bar'},
            },
            artifacts = {
                "confs" : self.confs,
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

        download_artifact(step.outputs.artifacts["labeled_data"])
        download_artifact(step.outputs.artifacts["logs"])

        for ii in jsonpickle.decode(step.outputs.parameters['task_names'].value):
            self.check_run_vasp_output(ii)

        # for ii in range(6):
        #     self.check_run_vasp_output(f'task.{ii:06d}')
            
