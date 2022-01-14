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
from mocked_ops import (
    MockedCollectData,
)


class TestMockedCollectData(unittest.TestCase):
    def setUp(self): 
        self.name = 'outdata'
        self.labeled_data = ['d0', 'd1']
        self.labeled_data = [Path(ii) for ii in self.labeled_data]
        for ii in self.labeled_data:
            (ii).mkdir(exist_ok=True, parents=True)
            (ii/'data').write_text(f'data of {str(ii)}')

    def tearDown(self):
        for ii in ['d0', 'd1', 'outdata']:
            ii=Path(ii)
            if ii.is_dir():
                shutil.rmtree(ii)
    
    def test(self):
        op = MockedCollectData()
        out = op.execute(OPIO({
            'name': self.name,
            'labeled_data' : self.labeled_data,
        }))
        out_data = out['labeled_data']

        self.assertTrue(out_data.is_dir())
        self.assertTrue((out_data/'d0').is_dir())
        self.assertTrue((out_data/'d1').is_dir())
        self.assertTrue((out_data/'d0'/'data').read_text(), 'data of d0')
        self.assertTrue((out_data/'d1'/'data').read_text(), 'data of d1')
        
        
class TestMockedCollectDataArgo(unittest.TestCase):
    def setUp(self):
        self.name = 'outdata'
        self.labeled_data = ['d0', 'd1']
        self.labeled_data = [Path(ii) for ii in self.labeled_data]
        for ii in self.labeled_data:
            (ii).mkdir(exist_ok=True, parents=True)
            (ii/'data').write_text(f'data of {str(ii)}')
        self.labeled_data = upload_artifact(self.labeled_data)

    def tearDown(self):
        for ii in ['d0', 'd1', 'outdata']:
            ii=Path(ii)
            if ii.is_dir():
                shutil.rmtree(ii)
        
    def test(self):
        coll_data = Step(
            'coll-data', 
            template = PythonOPTemplate(
                MockedCollectData,
                image = 'dflow:v1.0',
                output_artifact_archive={
                    "labeled_data" : None,
                },
                python_packages = '../dpgen2',
            ),
            parameters = {
                "name" : self.name,
            },
            artifacts = {
                "labeled_data" : self.labeled_data,
            },
        )        

        wf = Workflow(name="coll")
        wf.add(coll_data)
        wf.submit()

        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(2)

        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step(name="coll-data")[0]
        self.assertEqual(step.phase, "Succeeded")

        download_artifact(step.outputs.artifacts["labeled_data"])

        out_data = Path(self.name)
        self.assertTrue(out_data.is_dir())
        self.assertTrue((out_data/'d0').is_dir())
        self.assertTrue((out_data/'d1').is_dir())
        self.assertTrue((out_data/'d0'/'data').read_text(), 'data of d0')
        self.assertTrue((out_data/'d1'/'data').read_text(), 'data of d1')
        
        
