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
    Artifact
)

import time, shutil
from typing import Set, List
from pathlib import Path

from context import dpgen2
from dpgen2.op.run_dp_train import MockRunDPTrain
from dpgen2.op.prep_dp_train import MockPrepDPTrain
from dpgen2.flow.train_dp import steps_train


def _check_log(
        tcase,
        fname, 
        path,
        script,
        init_model,
        init_data,
        iter_data,
):
    with open(fname) as fp:
        lines = fp.read().strip().split('\n')
    tcase.assertEqual(
        lines[0].split(' '),
        ['init_model', str(Path(path)/init_model), 'OK']
    )
    for ii in range(2):
        tcase.assertEqual(
            lines[1+ii].split(' '),
            ['data', str(Path(path)/sorted(list(init_data))[ii]), 'OK']
        )
    for ii in range(2):
        tcase.assertEqual(
            lines[3+ii].split(' '),
            ['data', str(Path(path)/sorted(list(iter_data))[ii]), 'OK']
        )
    tcase.assertEqual(
        lines[5].split(' '),
        ['script', str(Path(path)/script), 'OK']
    )
    

def _check_model(
        tcase,
        fname,
        path,
        model,
):
    with open(fname) as fp:
        flines = fp.read().strip().split('\n')
    with open(Path(path)/model) as fp:
        mlines = fp.read().strip().split('\n')
    tcase.assertEqual(flines[0], "read from init model: ")
    for ii in range(len(mlines)):
        tcase.assertEqual(flines[ii+1], mlines[ii])

def _check_lcurve(
        tcase,
        fname,
        path,
        script,
):
    with open(fname) as fp:
        flines = fp.read().strip().split('\n')
    with open(Path(path)/script) as fp:
        mlines = fp.read().strip().split('\n')
    tcase.assertEqual(flines[0], "read from train_script: ")
    for ii in range(len(mlines)):
        tcase.assertEqual(flines[ii+1], mlines[ii])

def check_run_train_dp_output(
        tcase,
        work_dir, 
        script, 
        init_model,
        init_data,
        iter_data,
):
    cwd = os.getcwd()
    os.chdir(work_dir)    
    _check_log(tcase, "log", cwd, script, init_model, init_data, iter_data)
    _check_model(tcase, "model.pb", cwd, init_model)
    _check_lcurve(tcase, "lcurve.out", cwd, script)
    os.chdir(cwd)
    

class TestMockedPrepDPTrain(unittest.TestCase):
    def setUp(self):
        self.numb_models = 3
        self.template_script = { 'seed' : 1024, 'data': [] }
        self.expected_subdirs = ['task.0000', 'task.0001', 'task.0002']
        self.expected_train_scripts = [Path('task.0000/input.json'), Path('task.0001/input.json'), Path('task.0002/input.json')]

    def tearDown(self):
        for ii in self.expected_subdirs:
            if Path(ii).exists():
                shutil.rmtree(ii)

    def test(self):
        prep = MockPrepDPTrain()
        ip = OPIO({
            "template_script" : self.template_script,
            "numb_models" : self.numb_models,
        })
        op = prep.execute(ip)
        self.assertEqual(self.expected_train_scripts, op["train_scripts"])
        self.assertEqual(self.expected_subdirs, op["task_subdirs"])
        

class TestMockRunDPTrain(unittest.TestCase):
    def setUp(self):
        self.numb_models = 3

        tmp_models = []
        for ii in range(self.numb_models):
            ff = Path(f'model_{ii}.pb')
            ff.write_text(f'This is model {ii}')
            tmp_models.append(ff)
        self.init_models = tmp_models
        
        tmp_init_data = [Path('init_data/foo'), Path('init_data/bar')]
        for ii in tmp_init_data:
            ii.mkdir(exist_ok=True, parents=True)
            (ii/'a').write_text('data a')
            (ii/'b').write_text('data b')
        self.init_data = set(tmp_init_data)

        tmp_iter_data = [Path('iter_data/foo'), Path('iter_data/bar')]
        for ii in tmp_iter_data:
            ii.mkdir(exist_ok=True, parents=True)
            (ii/'a').write_text('data a')
            (ii/'b').write_text('data b')
        self.iter_data = set(tmp_iter_data)

        self.template_script = { 'seed' : 1024, 'data': [] }

        self.task_subdirs = ['task.0000', 'task.0001', 'task.0002']
        self.train_scripts = [Path('task.0000/input.json'), Path('task.0001/input.json'), Path('task.0002/input.json')]
        
        Path(self.task_subdirs[0]).mkdir(exist_ok=True, parents=True)
        Path(self.train_scripts[0]).write_text('{}')


    def tearDown(self):
        for ii in ['init_data', 'iter_data', self.task_subdirs[0]]:
            if Path(ii).exists():
                shutil.rmtree(str(ii))
        for ii in self.init_models:
            if Path(ii).exists():
                os.remove(ii)

    def test(self):
        run = MockRunDPTrain()
        ip = OPIO({
            "task_subdir": self.task_subdirs[0],
            "train_script": self.train_scripts[0],
            "init_model" : self.init_models[0],
            "init_data" : self.init_data,
            "iter_data" : self.iter_data,            
        })
        op = run.execute(ip)
        self.assertEqual(op["model"], Path('task.0000/model.pb'))
        self.assertEqual(op["log"], Path('task.0000/log'))
        self.assertEqual(op["lcurve"], Path('task.0000/lcurve.out'))
        check_run_train_dp_output(
            self, 
            self.task_subdirs[0], 
            self.train_scripts[0], 
            self.init_models[0], 
            self.init_data, 
            self.iter_data
        )


class TestTrainDp(unittest.TestCase):
    def setUp (self) :
        self.numb_models = 3

        tmp_models = []
        for ii in range(self.numb_models):
            ff = Path(f'model_{ii}.pb')
            ff.write_text(f'This is model {ii}')
            tmp_models.append(ff)
        self.init_models = upload_artifact(tmp_models)
        
        tmp_init_data = [Path('init_data/foo'), Path('init_data/bar')]
        for ii in tmp_init_data:
            ii.mkdir(exist_ok=True, parents=True)
            (ii/'a').write_text('data a')
            (ii/'b').write_text('data b')
        self.init_data = upload_artifact(tmp_init_data)

        tmp_iter_data = [Path('iter_data/foo'), Path('iter_data/bar')]
        for ii in tmp_iter_data:
            ii.mkdir(exist_ok=True, parents=True)
            (ii/'a').write_text('data a')
            (ii/'b').write_text('data b')
        self.iter_data = upload_artifact(tmp_iter_data)

        self.template_script = { 'seed' : 1024, 'data': [] }


    def test_train(self):
        steps = steps_train(
            "train-steps",
            # self.numb_models,
            # self.template_script,
            # self.init_models,
            # self.init_data,
            # self.iter_data, 
            MockPrepDPTrain,
            MockRunDPTrain,
        )
        train_step = Step(
            'train-step', 
            template = steps,
            parameters = {
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
            },
            artifacts = {
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        wf = Workflow(name="dp-train")
        wf.add(train_step)
        wf.submit()
        
        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(4)

        assert(wf.query_status() == "Succeeded")
        step = wf.query_step(name="train-step")[0]
        assert(step.phase == "Succeeded")

        download_artifact(step.outputs.artifacts["outcar"])
        download_artifact(step.outputs.artifacts["log"])
