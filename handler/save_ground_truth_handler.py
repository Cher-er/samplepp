import os
import pickle
import pandas as pd

from handler.handler import Handler
from container.container import container
from workload.workload import Workload


class SaveGroundTruthHandler(Handler):
    """
    [get]
        ground_truth
    """
    def pre_workload(self, output_dir="ground_truth") -> None:
        self.output_dir = output_dir
        self.ground_truth: list[pd.DataFrame] = []
    
    def handle(self) -> None:
        ground_truth = container.get("ground_truth")
        self.ground_truth.append(ground_truth)
    
    def post_workload(self) -> None:
        args = container.get("args")
        workload: Workload = container.get("Workload")
        path = os.path.join(self.output_dir, args.dataset, f"{workload.name}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.ground_truth, f)
        del self.ground_truth