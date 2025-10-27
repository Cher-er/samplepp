import duckdb
import pandas as pd
import time
from typing import Dict, Optional

from handler.handler import Handler
from container.container import container
from query.query import Query
from dataset.dataset import Dataset


class SumApproximateQueryProcessingHandler(Handler):
    """
    [get]
        samples
        generatives
        group_counts
    [register]
        self.result_name : result
    """
    def __init__(self, result_name: str) -> None:
        self.result_name = result_name

    def handle(self) -> None:
        query: Query = container.get("query")
        groupby_attr = container.get("groupby_attr")
        dataset: Dataset = container.get("Dataset")
        samples: Optional[Dict[str, pd.DataFrame]] = container.get("samples")
        generatives: Optional[Dict[str, pd.DataFrame]] = container.get("generatives", {})
        group_counts = container.get("group_counts")

        sql = query.sql

        conn = duckdb.connect()
        for table, sample in samples.items():
            if table not in dataset._facts:
                conn.register(table, sample)
                continue
            sample_rate = dataset.get_sample_rate(table)
            sample_group_counts = sample.groupby(groupby_attr).size().to_dict()
            sample_group_counts = {k: (v / sample_rate) for k, v in sample_group_counts.items()}

            if (generative := generatives.get(table)) is not None:
                sample = pd.concat([sample, generative], ignore_index=True)
            conn.register(table, sample)
        start = time.perf_counter()
        result = conn.execute(sql.replace("SUM", "AVG")).fetch_df()
        end = time.perf_counter()
        process_latency = end - start
        conn.close()

        groups = result.iloc[:, 0]
        avgs = result.iloc[:, 1]

        def compute_sum(group, avg):
            if group in group_counts:
                return avg * group_counts[group]
            else:
                return avg * sample_group_counts.get(group, 0) / sample_rate
        
        sums = [compute_sum(g, a) for g, a in zip(groups, avgs)]

        result = pd.DataFrame({
            'group': groups,
            'sum': sums
        })

        container.register(self.result_name, result)
        container.register_or_update("process_latency", process_latency)
    
    def post_query(self) -> None:
        container.remove(self.result_name)
        container.try_remove("process_latency")