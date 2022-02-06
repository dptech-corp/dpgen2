from dpgen2.exploration.lmp_task_group import LmpTaskGroup
from abc import (
    ABC,
    abstractmethod,
)
from dpgen2.constants import (
    lmp_conf_name, 
    lmp_input_name,
    model_name_pattern,
)
from dpgen2.exploration.lmp_task_group import (
    LmpTaskGroup,
    LmpTask,
)
from typing import (
    List,
)
from .group import ExplorationGroup

class ExplorationStage():
    """
    The exploration stage.

    """

    def __init__(self):
        self.clear()

    def clear(self):
        """
        Clear all exploration group.

        """
        self.explor_groups = []

    def add_group(
            self,
            grp : ExplorationGroup,
    ):
        """
        Add an exploration group
        
        Parameters
        ----------
        grp: ExplorationGroup
            The added exploration group

        """
        self.explor_groups.append(grp)
        return self

    def make_lmp_task_group(
            self,
    )->LmpTaskGroup:
        """
        Make the LAMMPS task group.        

        Returns
        -------
        task_grp: LmpTaskGroup
            The returned lammps task group. The number of tasks is equal to
            the summation of task groups defined by all the exploration groups
            added to the stage.

        """

        lmp_task_grp = LmpTaskGroup()
        for ii in self.explor_groups:
            # lmp_task_grp.add_group(ii.make_lmp_task_group())
            lmp_task_grp += ii.make_lmp_task_group()
        return lmp_task_grp


