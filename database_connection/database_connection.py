import psycopg2
import yaml
import pandas as pd

from container.container import container


class DatabaseConnection:
    def __init__(self) -> None:
        with open("application.yml", 'r') as f:
            config = yaml.safe_load(f)
        self.database_user = config['database']['user']
        self.database_host = config['database']['host']
        self.database_port = config['database']['port']
    
    def initialize(self) -> None:
        from dataset.dataset import Dataset
        dataset: Dataset = container.get("Dataset")
        db_name = dataset.db_name
        self.conn = psycopg2.connect(
            dbname=db_name,
            user=self.database_user,
            host=self.database_host,
            port=self.database_port
        )
    
    def execute(self, sql) -> pd.DataFrame:
        cur = self.conn.cursor()
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        result = cur.fetchall()
        cur.close()

        result = pd.DataFrame(result, columns=columns)
        return result
    
    def finalize(self) -> None:
        self.conn.close()