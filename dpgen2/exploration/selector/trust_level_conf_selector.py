import dpdata
from abc import ABC, abstractmethod
from typing import Tuple, List, Set
from pathlib import Path
from . import ConfFilters
from . import ConfSelector
from . import TrustLevel
from dpgen2.exploration.report import ExplorationReport    

class TrustLevelConfSelector(ConfSelector):
    def __init__(
            self,
            trust_level,
            conf_filters : ConfFilters = None,
    ):
        self.trust_level = trust_level
        self.conf_filters = conf_filters

    def select (
            self,
            trajs : List[Path],
            model_devis : List[Path],
            traj_fmt : str = 'deepmd/npy',
            type_map : List[str] = None,
    ) -> Tuple[List[ Path ], ExplorationReport]:
        raise NotImplementedError
