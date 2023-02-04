from abc import (
    ABC,
    abstractmethod,
)
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Set,
    Tuple,
)

import dpdata

from dpgen2.exploration.report import (
    ExplorationReport,
)

from . import (
    ConfFilters,
)


class ConfSelector(ABC):
    """Select configurations from trajectory and model deviation files."""

    @abstractmethod
    def select(
        self,
        trajs: List[Path],
        model_devis: List[Path],
        type_map: Optional[List[str]] = None,
    ) -> Tuple[List[Path], ExplorationReport]:
        pass
