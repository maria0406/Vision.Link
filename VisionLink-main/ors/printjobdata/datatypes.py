from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ExternalPrintjob:
    printjob_number: int
    image_file: Optional[bytes]
    file_type: Optional[str]


@dataclass
class Printjob:
    printjob_number: int
    image_file: Optional[bytes]
    file_type: Optional[str]
    features: Optional[np.ndarray]


class PrintjobRepository(ABC):
    @abstractmethod
    def get(self, printjob_id: int) -> Printjob:
        pass

    @abstractmethod
    def add(self, printjob: Printjob) -> Printjob:
        pass

    @abstractmethod
    def get_all(self) -> List[Printjob]:
        pass

    @abstractmethod
    def get_all_printjob_features(self) -> Dict[int, np.ndarray]:
        pass
