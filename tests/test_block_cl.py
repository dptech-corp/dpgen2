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
from dpgen2.op.prep_lmp import PrepLmp
from dpgen2.flow.prep_run_dp_train import prep_run_dp_train
from dpgen2.flow.prep_run_lmp import prep_run_lmp
from dpgen2.flow.prep_run_fp import prep_run_fp
from dpgen2.flow.block import block_cl
from dpgen2.utils.lmp_task_group import LmpTask, LmpTaskGroup
from dpgen2.fp.vasp import VaspInputs

from mock import patch

from dpgen2.constants import (
    train_task_pattern,
    train_script_name,
    train_log_name,
    model_name_pattern,
    lmp_conf_name,
    lmp_input_name,
    lmp_traj_name,
    lmp_log_name,
    vasp_task_pattern,
)
from mocked_ops import (    
    mocked_template_script,
    mocked_numb_models,
    make_mocked_init_models,
    make_mocked_init_data,
    mocked_incar_template,
    mocked_numb_select,
    MockedPrepDPTrain,
    MockedRunDPTrain,    
    MockedRunLmp,
    MockedPrepVasp,
    MockedRunVasp,
    MockedSelectConfs,
    MockedConfSelector,
    MockedExplorationReport,
    MockedCollectData,
    MockedLmpTaskGroup,
)


class TestBlockCL(unittest.TestCase):
    def _setUp_ops(self):
        self.prep_run_dp_train_op = prep_run_dp_train(
            "prep-run-dp-train",
            MockedPrepDPTrain,
            MockedRunDPTrain,
        )
        self.prep_run_lmp_op = prep_run_lmp(
            "prep-run-lmp",
            PrepLmp,
            MockedRunLmp,
        )
        self.prep_run_fp_op = prep_run_fp(
            "prep-run-fp",
            MockedPrepVasp,
            MockedRunVasp,
        )

    def _setUp_data(self):
        self.numb_models = mocked_numb_models

        tmp_models = make_mocked_init_models(self.numb_models)
        self.init_models = upload_artifact(tmp_models)
        self.str_init_models = tmp_models

        tmp_init_data = make_mocked_init_data()
        self.init_data = upload_artifact(tmp_init_data)
        self.path_init_data = set(tmp_init_data)
        
        tmp_iter_data = [Path('iter-000'), Path('iter-001')]
        for ii in tmp_iter_data:
            ii.mkdir(exist_ok=True, parents=True)
            (ii/'a').write_text('data a')
            (ii/'b').write_text('data b')
        self.iter_data = upload_artifact(tmp_iter_data)
        self.path_iter_data = set(tmp_iter_data)
        
        self.template_script = mocked_template_script
        
        self.task_group_list = MockedLmpTaskGroup()

        self.conf_selector = MockedConfSelector()
        self.type_map = []

        self.incar = mocked_incar_template
        self.vasp_inputs = VaspInputs(
            self.incar,
            {'foo': 'bar'},
        )
        

    def setUp(self):
        self.name = 'iter-002'
        self._setUp_ops()
        self._setUp_data()
        self.block_cl = block_cl(
            self.name, 
            self.prep_run_dp_train_op,
            self.prep_run_lmp_op,
            MockedSelectConfs,
            self.prep_run_fp_op,
            MockedCollectData,
        )        

    def tearDown(self):
        for ii in ['init_data', 'iter_data', 'iter-000', 'iter-001', 'models']:
            ii = Path(ii)
            if ii.is_dir():
                shutil.rmtree(ii)            
        for ii in range(self.numb_models):
            name = Path(model_name_pattern % ii)
            if name.is_file():
                os.remove(name)

    def test(self):
        block_step = Step(
            'step', 
            template = self.block_cl,
            parameters = {
                "block_id" : self.name,
                "type_map" : self.type_map,
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
                "train_config" : {},
                "lmp_task_grp" : self.task_group_list,
                "lmp_config" : {},
                "conf_selector" : self.conf_selector,
                'fp_inputs' : self.vasp_inputs,
                "fp_config" : {},
            },
            artifacts = {
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        wf = Workflow(name="block")
        wf.add(block_step)
        wf.submit()
        
        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(4)
        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step(name='step')[0]
        self.assertEqual(step.phase, "Succeeded")        

        report = jsonpickle.decode(step.outputs.parameters['exploration_report'].value)
        download_artifact(step.outputs.artifacts["iter_data"], path = 'iter_data')
        download_artifact(step.outputs.artifacts["models"], path = Path('models')/self.name)
        
        # we know number of selected data is 2
        # by MockedConfSelector
        for ii in range(mocked_numb_select):
            task_name = vasp_task_pattern % ii
            self.assertEqual(
                '\n'.join([f'labeled_data of {task_name}',
                           f'select conf.{ii}',
                           f'mocked conf {ii}',
                           f'mocked input {ii}',
                           ]),
                (Path('iter_data')/self.name/('data_'+task_name)/'data').read_text()
            )
        for ii in self.path_iter_data:
            dname = Path('iter_data')/ii
            self.assertEqual((dname/'a').read_text(), 'data a')
            self.assertEqual((dname/'b').read_text(), 'data b')

        # new model is read from init model
        for ii in range(self.numb_models):
            model = Path('models')/self.name/(train_task_pattern%ii)/'model.pb'
            flines = model.read_text().strip().split('\n')
            self.assertEqual(flines[0], "read from init model: ")
            self.assertEqual(flines[1], f"This is init model {ii}")
        
        # check report
        self.assertEqual(report.accurate_ratio(), .8)
        self.assertEqual(report.failed_ratio(), 0.1)
        self.assertEqual(report.candidate_ratio(), 0.1)
