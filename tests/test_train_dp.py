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

from typing import Set, List
from pathlib import Path

from context import dpgen2
from dpgen2.op.run_dp_train import MockRunDPTrain
from dpgen2.op.prep_dp_train import MockPrepDPTrain
from dpgen2.flow.train_dp import steps_train

class TestTrainDp(unittest.TestCase):
    def setUp (self) :
        self.numb_models = 3

        tmp_models = []
        for ii in range(self.numb_models):
            ff = Path(f'model_{ii}.pb')
            ff.write_text(f'This is model {ii}')
            tmp_models.append(ff)
        self.init_model = upload_artifact(tmp_models)
        
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
            self.numb_models,
            self.template_script,
            self.init_model,
            self.init_data,
            self.iter_data, 
            MockPrepDPTrain,
            MockRunDPTrain,
        )
        wf = Workflow(name="dp-train", steps=steps)
        wf.submit()
        
        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(4)

        assert(wf.query_status() == "Succeeded")
        step = wf.query_step(name="dp-train")[0]
        assert(step.phase == "Succeeded")
        download_artifact(step.outputs.artifacts["outcar"])
        download_artifact(step.outputs.artifacts["log"])
