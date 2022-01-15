import itertools
from dpgen2.utils.lmp_task_group import LmpTaskGroup
from abc import (
    ABC,
    abstractmethod,
)
from .lmp import make_lmp_input
from dpgen2.constants import (
    lmp_conf_name, 
    lmp_input_name,
    model_name_pattern,
)
from dpgen2.utils.lmp_task_group import (
    LmpTaskGroup,
    LmpTask,
)
from typing import (
    List,
)

class ExplorationGroup(ABC):
    @abstractmethod
    def make_lmp_task_group(
            self,
    )->LmpTaskGroup:
        pass

class CPTGroup(ExplorationGroup):
    def __init__(
            self,
            numb_models,
            mass_map,
            confs : List[str],
            temps : List[float],
            press : List[float] = None,
            ens : str = 'npt',
            dt : float = 0.001,
            nsteps : int = 1000,
            trj_freq : int = 10,
            tau_t : float = 0.1,
            tau_p : float = 0.5,
            pka_e : float = None,
            neidelay : int = None,
            no_pbc : bool = False,
            use_clusters : bool = False,
            relative_f_epsilon : float = None,
            relative_v_epsilon : float = None,
            ele_temp_f : float = None,
            ele_temp_a : float = None,
    ):
        self.graphs = [model_name_pattern % ii for ii in range(numb_models)]
        self.mass_map = mass_map
        self.confs = confs
        self.temps = temps
        self.press = press if press is not None else [None]
        self.ens = ens
        self.dt = dt
        self.nsteps = nsteps
        self.trj_freq = trj_freq
        self.tau_t = tau_t
        self.tau_p = tau_p
        self.pka_e = pka_e
        self.neidelay = neidelay
        self.no_pbc = no_pbc
        self.use_clusters = use_clusters
        self.relative_f_epsilon = relative_f_epsilon
        self.relative_v_epsilon = relative_v_epsilon
        self.ele_temp_f = ele_temp_f
        self.ele_temp_a = ele_temp_a

    def _make_lmp_task(
            self,
            conf : str,
            tt : float,
            pp : float,
    ) -> LmpTask:
        task = LmpTask()
        task\
            .add_file(
                lmp_conf_name, 
                conf,
            )\
            .add_file(
                lmp_input_name,
                make_lmp_input(
                    lmp_conf_name,
                    self.ens,
                    self.graphs,
                    self.nsteps,
                    self.dt,
                    self.neidelay,
                    self.trj_freq,
                    self.mass_map,
                    tt,
                    self.tau_t,
                    pp,
                    self.tau_p,
                    self.use_clusters,
                    self.relative_f_epsilon,
                    self.relative_v_epsilon,
                    self.pka_e,
                    self.ele_temp_f,
                    self.ele_temp_a,
                    self.no_pbc,
                )
            )
        return task

    def make_lmp_task_group(
            self,
    )->LmpTaskGroup:
        group = LmpTaskGroup()
        for cc,tt,pp in itertools.product(self.confs, self.temps, self.press):
            group.add_task(self._make_lmp_task(cc, tt, pp))
        return group
