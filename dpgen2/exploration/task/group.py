from abc import (
    ABC,
    abstractmethod,
)
from . import (
    ExplorationTaskGroup,
    ExplorationTask,
)
from typing import (
    List,
)

class ExplorationGroup(ABC):
    @abstractmethod
    def make_task(
            self,
    )->ExplorationTaskGroup:
        """
        Make the LAMMPS task group.
        """
        pass

