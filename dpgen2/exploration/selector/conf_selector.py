import dpdata
from abc import ABC, abstractmethod
from typing import Tuple, List, Set
from pathlib import Path
from . import (
    ConfFilters,
    TrustLevel,
)
from dpgen2.exploration.report import ExplorationReport

class ConfSelector(ABC):
    """Select configurations from trajectory and model deviation files.
    """
    @abstractmethod
    def select (
            self,
            trajs : List[Path],
            model_devis : List[Path],
            traj_fmt : str = 'deepmd/npy',
            type_map : List[str] = None,
    ) -> Tuple[List[ Path ], ExplorationReport]:
        pass


