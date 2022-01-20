import os, textwrap
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
from dflow.python import (
    FatalError,
)
from dpgen2.exploration.scheduler import (
    ConstTrustLevelStageScheduler,
    ExplorationScheduler,
)
from dpgen2.op.prep_lmp import PrepLmpTaskGroup
from dpgen2.flow.prep_run_dp_train import prep_run_dp_train
from dpgen2.flow.prep_run_lmp import prep_run_lmp
from dpgen2.flow.prep_run_fp import prep_run_fp
from dpgen2.flow.block import block_cl
from dpgen2.utils.lmp_task_group import LmpTask, LmpTaskGroup
from dpgen2.fp.vasp import VaspInputs
from dpgen2.exploration.stage import ExplorationStage
from dpgen2.exploration.report import ExplorationReport
from dpgen2.flow.schedule import dpgen, loop
from dpgen2.utils.lmp_task_group import LmpTaskGroup
from dpgen2.utils.trust_level import TrustLevel
from dpgen2.utils.conf_selector import TrustLevelConfSelector

from dpgen2.constants import (
    train_task_pattern,
    train_script_name,
    train_log_name,
    model_name_pattern,
    lmp_conf_name,
    lmp_input_name,
    lmp_traj_name,
    lmp_log_name,
)
from mocked_ops import (
    mocked_template_script,
    mocked_numb_models,
    make_mocked_init_models,
    make_mocked_init_data,
    MockedPrepDPTrain,
    MockedRunDPTrain,    
    MockedRunLmp,
    MockedPrepVasp,
    MockedRunVasp,
    MockedSelectConfs,
    MockedConfSelector,
    MockedExplorationReport,
    MockedCollectData,
    MockedExplorationReport,
    MockedLmpTaskGroup,
    MockedLmpTaskGroup1,
    MockedStage,
    MockedStage1,
    MockedConstTrustLevelStageScheduler,
)


class TestLoop(unittest.TestCase):
    def _setUp_ops(self):
        self.prep_run_dp_train_op = prep_run_dp_train(
            "prep-run-dp-train",
            MockedPrepDPTrain,
            MockedRunDPTrain,
        )
        self.prep_run_lmp_op = prep_run_lmp(
            "prep-run-lmp",
            PrepLmpTaskGroup,
            MockedRunLmp,
        )
        self.prep_run_fp_op = prep_run_fp(
            "prep-run-fp",
            MockedPrepVasp,
            MockedRunVasp,
        )
        self.block_cl_op = block_cl(
            self.name+'-block', 
            self.prep_run_dp_train_op,
            self.prep_run_lmp_op,
            MockedSelectConfs,
            self.prep_run_fp_op,
            MockedCollectData,
        )        
        self.loop_op = loop(
            self.name+'-loop',
            self.block_cl_op,
        )
        self.dpgen_op = dpgen(
            self.name,
            self.loop_op,
        )

    def _setUp_data(self):
        self.numb_models = 3

        tmp_models = []
        for ii in range(self.numb_models):
            ff = Path(model_name_pattern % ii)
            ff.write_text(f'This is init model {ii}')
            tmp_models.append(ff)
        self.init_models = upload_artifact(tmp_models)
        self.str_init_models = tmp_models

        tmp_init_data = [Path('init_data/foo'), Path('init_data/bar')]
        for ii in tmp_init_data:
            ii.mkdir(exist_ok=True, parents=True)
            (ii/'a').write_text('data a')
            (ii/'b').write_text('data b')
        self.init_data = upload_artifact(tmp_init_data)
        self.path_init_data = set(tmp_init_data)
        
        self.iter_data = upload_artifact([])
        self.path_iter_data = None
        
        self.template_script = { 'seed' : 1024, 'data': [] }        

        self.conf_filters = []
        self.type_map = []

        self.incar = 'incar template'
        self.vasp_inputs = VaspInputs(
            self.incar,
            {'foo': 'bar'},
        )


        self.scheduler = ExplorationScheduler()        
        self.trust_level = TrustLevel(0.1, 0.3)
        trust_level = TrustLevel(0.1, 0.3)
        stage_scheduler = MockedConstTrustLevelStageScheduler(
            MockedStage(),
            trust_level,
            conv_accuracy = 0.7,
            max_numb_iter = 2,
        )
        self.scheduler.add_stage_scheduler(stage_scheduler)
        trust_level = TrustLevel(0.2, 0.4)
        stage_scheduler = MockedConstTrustLevelStageScheduler(
            MockedStage1(),
            trust_level,
            conv_accuracy = 0.7,
            max_numb_iter = 2,
        )
        self.scheduler.add_stage_scheduler(stage_scheduler)        
        
        
    def setUp(self):
        self.name = 'dpgen'
        self._setUp_ops()
        self._setUp_data()

    def test(self):
        dpgen_step = Step(
            'dpgen-step', 
            template = self.dpgen_op,
            parameters = {
                "type_map" : self.type_map,
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
                "conf_filters" : self.conf_filters,
                'fp_inputs' : self.vasp_inputs,
                "exploration_scheduler" : self.scheduler,
            },
            artifacts = {
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        
        wf = Workflow(name="dpgen")
        wf.add(dpgen_step)
        wf.submit()
        
        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(4)
        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step(name='dpgen-step')[0]
        self.assertEqual(step.phase, "Succeeded")        
        
        
