from abc import ABC, abstractmethod
import dpdata

class ConfFilter(ABC):
    @abstractmethod
    def check (
            self,
            conf : dpdata.System,
    ) -> bool :
        pass
