from abc import ABC, abstractmethod

from .conf_filter import ConfFilters
from .trust_level import TrustLevel
from typing import Tuple, List, Set
from pathlib import Path

class ConfSelector(ABC):

    @abstractmethod
    def select (
            self,
            trajs : List[Path],
            model_devis : List[Path],
            traj_fmt : str = 'deepmd/npy',
            type_map : List[str] = None,
    ) -> Tuple[List[ Path ], TrustLevel]:
        pass


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
    ) -> Tuple[List[ Path ], TrustLevel]:
        raise NotImplementedError
