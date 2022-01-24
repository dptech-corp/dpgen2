from __future__ import annotations
from abc import ABC, abstractmethod
import dpdata

class ConfFilter(ABC):
    @abstractmethod
    def check (
            self,
            conf : dpdata.System,
    ) -> bool :
        pass

class ConfFilters():
    def __init__(
            self,
    ):
        self._filters = []

    def add(
            self,
            conf_filter : ConfFilter,
    )->ConfFilters:
        self._filters.append(conf_filter)
        return self

    def check(
            self,
            conf : dpdata.System,
    ) -> bool : 
        return all([ ii.check(conf) for ii in self._filters ])
    
