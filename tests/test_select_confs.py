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
from typing import Set, List, Tuple
from pathlib import Path

try:
    from context import dpgen2
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from dpgen2.op.select_confs import SelectConfs
from dpgen2.utils.conf_selector import ConfSelector
from dpgen2.utils.conf_filter import ConfFilter
from dpgen2.utils.exploration_report import ExplorationReport

upload_packages.append(__file__)

class MockedExplorationReport(ExplorationReport):
    def __init__(self):
        pass

    def converged (
            self, 
    ) -> bool :
        return True

    def failed_ratio (
            self, 
            tag = None,
    ) -> float :
        return 0.

    def accurate_ratio (
            self,
            tag = None,
    ) -> float :
        return 1.

    def candidate_ratio (
            self,
            tag = None,
    ) -> float :
        return 0.

    def update_trust_levels (
            self,
    ) -> Tuple[float] :
        return (0.1, 0.2, 0.1, 0.2)


class MockedConfSelector(ConfSelector):
    def select (
            self,
            trajs : List[Path],
            model_devis : List[Path],
            conf_filters : List[ConfFilter] = [],
            traj_fmt : str = 'deepmd/npy',
            type_map : List[str] = None,
    ) -> List[ Path ] :
        confs = []
        fname = Path('conf.0')
        fname.write_text('conf of conf.0')
        confs.append(fname)
        fname = Path('conf.1')
        fname.write_text('conf of conf.1')
        confs.append(fname)
        return confs

class MockedSelectConfs(SelectConfs):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        conf_selector = ip['conf_selector']
        trajs = ip['trajs']
        model_devis = ip['model_devis']
        confs = conf_selector.select(trajs, model_devis)
        report = MockedExplorationReport()

        return OPIO({
            "report" : report,
            "confs" : confs,
        })


class TestMockedSelectConfs(unittest.TestCase):
    def setUp(self): 
        self.conf_selector = MockedConfSelector()
        self.conf_filters = []
        self.traj_fmt = 'foo'
        self.type_map = []
        self.trajs = [Path('traj.foo'), Path('traj.bar')]
        self.model_devis = [Path('md.foo'), Path('md.bar')]

    def tearDown(self):
        for ii in ['conf.0', 'conf.1']:
            ii=Path(ii)
            if ii.is_file():
                os.remove(ii)
    
    def test(self):
        op = MockedSelectConfs()
        out = op.execute(OPIO({
            'conf_selector': self.conf_selector,
            'conf_filters': self.conf_filters,
            'traj_fmt': self.traj_fmt,
            'type_map' : self.type_map,
            'trajs' : self.trajs,
            'model_devis' : self.model_devis,
        }))
        confs = out['confs']
        report = out['report']

        self.assertTrue(report.converged())
        self.assertTrue(confs[0].is_file())
        self.assertTrue(confs[1].is_file())
        self.assertTrue(confs[0].read_text(), 'conf of conf.0')
        self.assertTrue(confs[1].read_text(), 'conf of conf.1')
        
        
        
