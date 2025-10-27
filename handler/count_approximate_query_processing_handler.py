import pandas as pd
import time

from handler.handler import Handler
from container.container import container


class CountApproximateQueryProcessingHandler(Handler):
    """
    [get]
        group_counts
    [register]
        self.result_name : result
    """
    def __init__(self, result_name: str) -> None:
        self.result_name = result_name

    def handle(self) -> None:
        group_counts = container.get("group_counts")
        start = time.perf_counter()
        result = pd.DataFrame.from_dict(group_counts, orient='index').reset_index()
        end = time.perf_counter()
        process_latency = end - start
        container.register(self.result_name, result)
        container.register_or_update("process_latency", process_latency)
    
    def post_query(self) -> None:
        container.remove(self.result_name)
        container.try_remove("process_latency")