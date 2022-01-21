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

    @abstractmethod
    def plan_next_iteration(
            self,
            report : ExplorationReport,
            confs : List[Path],
    ) -> Tuple[bool, LmpTaskGroup, ConfSelector] :
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
        self.stage_schedulers.append(stage_scheduler)
        return self

    def get_stage(self):
        return self.cur_stage

    def get_iteration(self):
        return self.iteration

    def plan_next_iteration(
            self,
            report : ExplorationReport = None,
            trajs : List[Path] = None,
    ) -> Tuple[bool, LmpTaskGroup, ConfSelector] :

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
                # self.iteration += 1
                return True, None, None,
        else :
            self.iteration += 1
            return converged, lmp_task_grp, conf_selector

