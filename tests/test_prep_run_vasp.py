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
)

import time, shutil, json, jsonpickle
from typing import Set, List
from pathlib import Path
try:
    from context import dpgen2
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from dpgen2.flow.prep_run_fp import prep_run_fp
from mocked_ops import (
    MockedPrepVasp,
    MockedRunVasp,
)
from dpgen2.fp.vasp import VaspInputs

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
            'inputs' : \
            VaspInputs(
                self.incar,
                {'foo': 'bar'}
            ),
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
                'inputs' : \
                VaspInputs(
                    self.incar,
                    {'foo': 'bar'}
                ),
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
            
