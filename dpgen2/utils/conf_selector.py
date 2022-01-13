from abc import ABC, abstractmethod

from .conf_filter import ConfFilter
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
    ) -> List[ Path ] :
        pass
