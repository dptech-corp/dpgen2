from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
import os, json
from typing import Tuple, List, Set
from pathlib import Path
from dpgen2.utils.conf_selector import ConfSelector
from dpgen2.utils.conf_filter import ConfFilter
from dpgen2.utils.exploration_report import ExplorationReport

class SelectConfs(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "conf_selector": ConfSelector,
            "conf_filters": List[ConfFilter],
            "traj_fmt": str,
            "type_map": List[str],

            "trajs": Artifact(List[Path]),
            "model_devis": Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "report" : ExplorationReport,

            "confs" : Artifact(List[Path]),
        })
