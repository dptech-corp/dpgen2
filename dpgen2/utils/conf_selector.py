from abc import ABC, abstractmethod

from .conf_filter import ConfFilter
from .trust_level import TrustLevel
from typing import Tuple, List, Set
from pathlib import Path

class ConfSelector(ABC):

    @abstractmethod
    def select (
            self,
            trajs : List[Path],
            model_devis : List[Path],
            conf_filters : List[ConfFilter] = [],
            traj_fmt : str = 'deepmd/npy',
            type_map : List[str] = None,
    ) -> Tuple[List[ Path ], TrustLevel]:
        pass


class TrustLevelConfSelector(ConfSelector):
    def __init__(
            self,
            trust_level,
    ):
        self.trust_level = trust_level

    def select (
            self,
            trajs : List[Path],
            model_devis : List[Path],
            conf_filters : List[ConfFilter] = [],
            traj_fmt : str = 'deepmd/npy',
            type_map : List[str] = None,
    ) -> Tuple[List[ Path ], TrustLevel]:
        raise NotImplementedError
