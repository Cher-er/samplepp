import duckdb
import pandas as pd
import time
from typing import Dict, Optional

from handler.handler import Handler
from container.container import container
from query.query import Query


class AvgApproximateQueryProcessingHandler(Handler):
    """
    [get]
        samples
        generatives
    [register]
        self.result_name : result
    """
    def __init__(self, result_name: str) -> None:
        self.result_name = result_name

    def handle(self) -> None:
        query: Query = container.get("query")
        samples: Optional[Dict[str, pd.DataFrame]] = container.get("samples")
        generatives: Optional[Dict[str, pd.DataFrame]] = container.get("generatives", {})

        sql = query.sql

        conn = duckdb.connect()
        for table, sample in samples.items():
            if (generative := generatives.get(table)) is not None:
                sample = pd.concat([sample, generative], ignore_index=True)
            conn.register(table, sample)
        start = time.perf_counter()
        result = conn.execute(sql).fetch_df()
        end = time.perf_counter()
        process_latency = end - start
        conn.close()

        container.register(self.result_name, result)
        container.register_or_update("process_latency", process_latency)
    
    def post_query(self) -> None:
        container.remove(self.result_name)
        container.try_remove("process_latency")