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


# class NativeExplorationReport(ExplorationReport):
#     def __init__(
#             self, 
#             trust_f_lo,
#             trust_f_hi,
#             trust_v_lo = None,
#             trust_v_hi = None,
#     ):
#         super().__init__()
#         reset_counters()
#         self.trust_f_lo = trust_f_lo

#     def reset_counters(self):
#         self.counters = { "_global" : Counter()}
        
#     def add_frame(
#             self, 
            
#             tag = None,
#     ):
        
