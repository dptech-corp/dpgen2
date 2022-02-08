from typing import (
    List,
    Tuple,
)
from dflow.python import (
    FatalError,
)
from pathlib import Path
from dpgen2.exploration.report import ExplorationReport
from dpgen2.exploration.task import ExplorationTaskGroup, ExplorationStage
from dpgen2.exploration.selector import ConfSelector, TrustLevelConfSelector, TrustLevel
from . import StageScheduler

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
    ) -> Tuple[bool, ExplorationTaskGroup, TrustLevelConfSelector] :
        if report is None:
            converged = False
            lmp_task_grp = self.stage.make_task()
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
                lmp_task_grp = self.stage.make_task()
                ret_selector = self.selector
        self.nxt_iter += 1
        return converged, lmp_task_grp, ret_selector
