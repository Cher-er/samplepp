import re
import pandas as pd

from handler.ground_truth_handler import GroundTruthHandler
from container.container import container


class GroundTruthSampleHandler(GroundTruthHandler):
    """
    [register]
        ground_truth_sample
    """

    def handle(self) -> None:
        query = container.get("query")
        sql = query.sql
        self.get_ground_truth_sample(sql)
    
    def get_ground_truth_sample(self, sql) -> None:
        db_conn = container.get("DatabaseConnection")
        from_match = re.search(r'\bFROM\b\s+([^\s]+)', sql, re.IGNORECASE)
        where_match = re.search(r'\bWHERE\b\s+(.+?)\s+GROUP BY', sql, re.IGNORECASE)
        
        table = from_match.group(1)
        where_clause = where_match.group(1) if where_match else None

        sql = f"SELECT * FROM {table}"
        if where_clause:
            sql += f" WHERE {where_clause.strip()}"

        ground_truth_sample: pd.DataFrame = db_conn.execute(sql)
        container.register("ground_truth_sample", ground_truth_sample)
    
    def post_query(self) -> None:
        container.remove("ground_truth_sample")