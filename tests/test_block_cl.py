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
from dpgen2.flow.prep_run_dp_train import prep_run_dp_train
from dpgen2.flow.prep_run_lmp import prep_run_lmp
from dpgen2.flow.prep_run_fp import prep_run_fp
from dpgen2.flow.block import block_cl
from dpgen2.utils.lmp_task_group import LmpTask, LmpTaskGroup
from dpgen2.fp.vasp import VaspInputs

from mocked_ops import (    
    MockedPrepDPTrain,
    MockedRunDPTrain,    
    MockedRunLmp,
    MockedPrepVasp,
    MockedRunVasp,
    MockedSelectConfs,
    MockedConfSelector,
    MockedExplorationReport,
    MockedCollectData,
)

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


class TestBlockCL(unittest.TestCase):
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

    def _setUp_data(self):
        self.numb_models = 3

        tmp_models = []
        for ii in range(self.numb_models):
            ff = Path(f'model_{ii}.pb')
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
        
        tmp_iter_data = [Path('iter-000'), Path('iter-001')]
        for ii in tmp_iter_data:
            ii.mkdir(exist_ok=True, parents=True)
            (ii/'a').write_text('data a')
            (ii/'b').write_text('data b')
        self.iter_data = upload_artifact(tmp_iter_data)
        self.path_iter_data = set(tmp_iter_data)
        
        self.template_script = { 'seed' : 1024, 'data': [] }
        
        self.ngrp = 2
        self.ntask_per_grp = 3
        self.task_group_list = make_task_group_list(self.ngrp, self.ntask_per_grp)

        self.conf_selector = MockedConfSelector()
        self.conf_filters = []
        self.type_map = []

        self.incar = 'incar template'
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
            name = Path(f'model_{ii}.pb')
            if name.is_file():
                os.remove(name)

    def test(self):
        block_step = Step(
            'step', 
            template = self.block_cl,
            parameters = {
                "type_map" : self.type_map,
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
                "lmp_task_grps" : self.task_group_list,
                "conf_selector" : self.conf_selector,
                "conf_filters" : self.conf_filters,
                'fp_inputs' : self.vasp_inputs,
            },
            artifacts = {
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        
        wf = Workflow(name="dpgen")
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
        for ii in range(2):
            task_name = f'task.{ii:06d}'
            self.assertEqual(
                f'labeled_data of {task_name}',
                (Path('iter_data')/self.name/('data_'+task_name)/'data').read_text())
        for ii in self.path_iter_data:
            dname = Path('iter_data')/ii
            self.assertEqual((dname/'a').read_text(), 'data a')
            self.assertEqual((dname/'b').read_text(), 'data b')

        # new model is read from init model
        for ii in range(self.numb_models):
            model = Path('models')/self.name/f'task.{ii:04d}'/'model.pb'
            flines = model.read_text().strip().split('\n')
            self.assertEqual(flines[0], "read from init model: ")
            self.assertEqual(flines[1], f"This is init model {ii}")
        
        # check report
        self.assertEqual(report.accurate_ratio(), 1.)
        self.assertEqual(report.failed_ratio(), 0.)
        self.assertEqual(report.candidate_ratio(), 0.)
