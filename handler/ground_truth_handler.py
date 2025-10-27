import pandas as pd

from handler.handler import Handler
from container.container import container
from database_connection.database_connection import DatabaseConnection

class GroundTruthHandler(Handler):
    """
    [register]
        ground_truth
    """
    def handle(self) -> None:
        query = container.get("query")
        sql = query.sql
        self.get_ground_truth(sql)

    def get_ground_truth(self, sql) -> None:
        db_conn: DatabaseConnection = container.get("DatabaseConnection")

        ground_truth: pd.DataFrame = db_conn.execute(sql)
        container.register("ground_truth", ground_truth)
    
    def post_query(self) -> None:
        container.remove("ground_truth")