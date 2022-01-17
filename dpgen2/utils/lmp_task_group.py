import os
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Sequence,
)
from typing import (
    List,
    Tuple,
    Dict,
)

class LmpTask():
    def __init__(
            self, 
    ):
        self._files = {}

    def add_file(
            self,
            fname, fcont,
    ):
        self._files[fname] = fcont
        return self

    def files(self) -> Dict:
        return self._files


class LmpTaskGroup(Sequence):
    def __init__(self):
        super().__init__()
        self._task_list = []

    def __getitem__(self, ii:int) -> LmpTask:
        return self.task_list[ii]

    def __len__(self) -> int:
        return len(self.task_list)

    @property
    def task_list(self) -> List[LmpTask]:
        return self._task_list

    def add_task(self, task):
        self.task_list.append(task)
        return self

    def add_group(
            self,
            group,
    ):
        self._task_list = self._task_list + group._task_list
        return self

    def __add__(
            self,
            group,
    ):        
        return self.add_group(group)


class FooTask(LmpTask):
    def __init__(
            self, 
            conf_name = 'conf.lmp', 
            conf_cont = '',
            inpu_name = 'in.lammps',
            inpu_cont = '',
    ):
        super().__init__()
        self._files = {
            conf_name : conf_cont,
            inpu_name : inpu_cont,
        }


class FooTaskGroup(LmpTaskGroup):
    def __init__(self, numb_task):
        super().__init__()
        self.tlist = []
        for ii in range(numb_task):
            self.tlist.add_task( 
                FooTask(f'conf.{ii}', f'this is conf.{ii}',
                        f'input.{ii}', f'this is input.{ii}',
                        )
            )

    @property
    def task_list(self):
        return self.tlist


if __name__ == '__main__':
    grp = FooTaskGroup(3)
    for ii in grp:
        fcs = ii.files()
        print(fcs)

