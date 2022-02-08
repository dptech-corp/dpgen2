from abc import ABC, abstractmethod
from typing import Tuple

class ExplorationReport(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def failed_ratio (
            self, 
            tag = None,
    ) -> float :
        pass

    @abstractmethod
    def accurate_ratio (
            self,
            tag = None,
    ) -> float :
        pass

    @abstractmethod
    def candidate_ratio (
            self,
            tag = None,
    ) -> float :
        pass


