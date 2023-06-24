from typing import Dict, List

import numpy as np

from ors.printjobdata.datatypes import Printjob, PrintjobRepository


class InMemoryPrintjobRepository(PrintjobRepository):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.store = {}

    def get(self, printjob_id: int) -> Printjob:
        return self.store[printjob_id]

    def add(self, printjob: Printjob) -> Printjob:
        self.store[printjob.printjob_number] = printjob

    def get_all(self) -> List[Printjob]:
        return list(self.store.values())

    def get_all_printjob_features(self) -> Dict[int, np.ndarray]:
        return {job_id: job.features for job_id, job in self.store.items()}
