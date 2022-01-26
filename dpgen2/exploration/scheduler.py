from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    List,
    Tuple,
)
from dflow.python import (
    FatalError,
)
from pathlib import Path
from .stage import ExplorationStage
from .report import ExplorationReport
from dpgen2.utils.lmp_task_group import LmpTaskGroup
from dpgen2.utils.conf_selector import ConfSelector, TrustLevelConfSelector
from dpgen2.utils.trust_level import TrustLevel

class StageScheduler(ABC):
    """
    The scheduler for an exploration stage.
    """

    @abstractmethod
    def plan_next_iteration(
            self,
            hist_reports : List[ExplorationReport] = [],
            report : ExplorationReport,
            confs : List[Path],
    ) -> Tuple[bool, LmpTaskGroup, ConfSelector] :
        """
        Make the plan for the next iteration of the stage.

        It checks the report of the current and all historical iterations of the stage, and tells if the iterations are converged. If not converged, it will plan the next ieration for the stage. 

        Parameters
        ----------
        hist_reports: List[ExplorationReport]
            The historical exploration report of the stage. If this is the first iteration of the stage, this list is empty.
        report : ExplorationReport
            The exploration report of this iteration.
        confs: List[Path]
            A list of configurations generated during the exploration. May be used to generate new configurations for the next iteration. 

        Returns
        -------
        converged: bool
            If the stage converged.
        lmp_task_group: LmpTaskGroup
            A `LmpTaskGroup` defining the exploration of the next iteration. Should be `None` if the stage is converged.
        conf_selector: ConfSelector
            The configuration selector for the next iteration. Should be `None` if the stage is converged.

        """
        pass


class ConstTrustLevelStageScheduler(StageScheduler):    
    def __init__(
            self,
            stage : ExplorationStage,
            trust_level : TrustLevel,
            conv_accuracy : float = 0.9,
            max_numb_iter : int = None,
    ):
        self.stage = stage
        self.trust_level = trust_level
        self.conv_accuracy = conv_accuracy
        self.selector = TrustLevelConfSelector(self.trust_level)
        self.nxt_iter = 0
        self.max_numb_iter = max_numb_iter

    def plan_next_iteration(
            self,
            hist_reports : List[ExplorationReport] = [],
            report : ExplorationReport = None,
            trajs : List[Path] = None,
    ) -> Tuple[bool, LmpTaskGroup, TrustLevelConfSelector] :
        if report is None:
            converged = False
            lmp_task_grp = self.stage.make_lmp_task_group()
            ret_selector = self.selector
        else :
            converged = report.accurate_ratio() >= self.conv_accuracy
            if converged:
                # if converged, no more lmp task
                lmp_task_grp = None
                ret_selector = None
            else :
                # if not converged, check max iter and make lmp tasks
                if self.max_numb_iter is not None and self.nxt_iter == self.max_numb_iter:
                    raise FatalError('reached maximal number of iterations')
                lmp_task_grp = self.stage.make_lmp_task_group()
                ret_selector = self.selector
        self.nxt_iter += 1
        return converged, lmp_task_grp, ret_selector


class ExplorationScheduler():
    """
    The exploration scheduler.

    """

    def __init__(
            self,
    ):
        self.stage_schedulers = []
        self.stage_reports = [[]]
        self.cur_stage = 0
        self.iteration = -1
        
    def add_stage_scheduler(
            self,
            stage_scheduler : StageScheduler,
    ):
        """
        Add stage scheduler. 

        All added schedulers can be treated as a `list` (order matters). Only one stage is converged, the iteration goes to the next iteration.

        Parameters
        ----------
        stage_scheduler: StageScheduler
            The added stage scheduler
        
        """
        self.stage_schedulers.append(stage_scheduler)
        return self

    def get_stage(self):
        """
        Get the index of current stage. 

        Stage index increases when the previous stage converges. Usually called after `self.plan_next_iteration`.

        """
        return self.cur_stage

    def get_iteration(self):
        """
        Get the index of the current iteration.

        Iteration index increase when `self.plan_next_iteration` returns valid `lmp_task_grp` and `conf_selector` for the next iteration.

        """
        return self.iteration

    def plan_next_iteration(
            self,
            report : ExplorationReport = None,
            trajs : List[Path] = None,
    ) -> Tuple[bool, LmpTaskGroup, ConfSelector] :
        """
        Make the plan for the next DPGEN iteration.

        Parameters
        ----------
        report : ExplorationReport
            The exploration report of this iteration.
        confs: List[Path]
            A list of configurations generated during the exploration. May be used to generate new configurations for the next iteration. 

        Returns
        -------
        converged: bool
            If DPGEN converges.
        lmp_task_group: LmpTaskGroup
            A `LmpTaskGroup` defining the exploration of the next iteration. Should be `None` if converged.
        conf_selector: ConfSelector
            The configuration selector for the next iteration. Should be `None` if converged.

        """

        try:
            converged, lmp_task_grp, conf_selector = \
                self.stage_schedulers[self.cur_stage].plan_next_iteration(
                    self.stage_reports[self.cur_stage],
                    report,
                    trajs,
                )
            self.stage_reports[self.cur_stage].append(report)
        except FatalError as e:
            raise FatalError(f'stage {self.cur_stage}: ' + str(e))
        
        if converged:
            self.cur_stage += 1
            self.stage_reports.append([])
            if self.cur_stage < len(self.stage_schedulers):
                # goes to next stage
                return self.plan_next_iteration()
            else:
                # all stages converged
                return True, None, None,
        else :
            self.iteration += 1
            return converged, lmp_task_grp, conf_selector

