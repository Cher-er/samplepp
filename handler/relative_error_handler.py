import logging
logger = logging.getLogger(__name__)

import numpy as np

from handler.handler import Handler
from container.container import container


class RelativeErrorHandler(Handler):
    """
    [get]
        ground_truth
        self.result_name
        groups
    [register]
        self.re_name : relative_error
    [register or update]
        precision
        recall
    """
    def __init__(self, re_name: str, result_name: str) -> None:
        self.re_name = re_name
        self.result_name = result_name

    def handle(self) -> None:
        ground_truth = container.get("ground_truth")
        result = container.get(self.result_name)

        truth_groups = set(ground_truth.iloc[:, 0])
        aqp_groups = set(result.iloc[:, 0])
        missing_groups = truth_groups - aqp_groups
        out_groups = aqp_groups - truth_groups
        if len(truth_groups) == 0:
            missing_rate = 0.0
            out_rate = 0.0
        else:
            missing_rate = len(missing_groups) / len(truth_groups)
            out_rate = len(out_groups) / len(truth_groups)
        
        if result is None or result.shape[0] == 0:
            relative_error = 1.0
            precision = 1.0
            recall = 0.0
        else:
            errors = {}
            ground_truth_dict = dict(zip(ground_truth.iloc[:, 0], ground_truth.iloc[:, 1]))
            result_dict = dict(zip(result.iloc[:, 0], result.iloc[:, 1]))
            all_group_keys = set(ground_truth_dict.keys()) | set(result_dict.keys())
            epsilon = 1e-8
            for group in all_group_keys:
                if group in ground_truth_dict and group in result_dict:
                    ground_truth_val = ground_truth_dict[group]
                    result_val = result_dict[group]
                    rel_error = abs(ground_truth_val - result_val) / (abs(ground_truth_val) + epsilon)
                    rel_error = min(1.0, rel_error)
                else:
                    rel_error = 1.0
                errors[group] = rel_error
            
            relative_error = np.mean(list(errors.values()))

            groups = container.get("groups")
            ground_truth_groups = set(ground_truth.iloc[:, 0])
            result_groups = set(result.iloc[:, 0])
            TP = len(ground_truth_groups & result_groups)
            FP = len(result_groups - ground_truth_groups)
            FN = len(ground_truth_groups - result_groups)
            TN = len(groups - (ground_truth_groups | result_groups))
            precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        container.register(self.re_name, relative_error)
        container.register_or_update("precision", precision)
        container.register_or_update("recall", recall)
        container.register_or_update("missing_rate", missing_rate)
        container.register_or_update("out_rate", out_rate)
    
    def post_query(self) -> None:
        container.remove(self.re_name)
        container.try_remove("precision")
        container.try_remove("recall")
        container.try_remove("missing_rate")
        container.try_remove("out_rate")