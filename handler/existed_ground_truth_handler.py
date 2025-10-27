import pickle
import pandas as pd

from handler.handler import Handler
from container.container import container


class ExistedGroundTruthHandler(Handler):
    """
    [register]
        ground_truth
    """
    def __init__(self, ground_truth_paths: list[str]) -> None:
        self.ground_truth_paths = ground_truth_paths
        self.workload_index = 0
        self.ground_truth: list[pd.DataFrame] = None
        self.query_index = None

    def pre_workload(self) -> None:
        ground_truth_path = self.ground_truth_paths[self.workload_index]
        with open(ground_truth_path, 'rb') as f:
            self.ground_truth = pickle.load(f)
        self.query_index = 0

    def handle(self) -> None:
        container.register("ground_truth", self.ground_truth[self.query_index])
    
    def post_query(self) -> None:
        container.remove("ground_truth")
        self.query_index += 1
    
    def post_workload(self) -> None:
        self.workload_index += 1