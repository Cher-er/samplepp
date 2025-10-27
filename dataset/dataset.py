import os
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union

from dataset.table import Table
from container.container import container
from container.initialize_level import initialize_level
from database_connection.database_connection import DatabaseConnection


def fact(facts: list[str]):
    def decorator(cls):
        cls._facts = facts
        return cls
    return decorator


@initialize_level(1)
class Dataset:
    tables: Dict[str, Table]
    db_name: str
    best_state_paths: dict[str, str]

    def initialize(self) -> None:
        for table_name, table in self.tables.items():
            df = self.get(table_name)
            schema = self.get_scehma(table_name)
            for col_name, col_type in schema.items():
                match col_type:
                    case "int":
                        table.uniques[col_name] = {item.item() for item in df[col_name].unique()}
                    case "str":
                        table.uniques[col_name] = {item for item in df[col_name].unique()}
                    case "num":
                        table.minmaxs[col_name] = (df[col_name].min().item(), df[col_name].max().item())
    
    @property
    def table_names(self) -> List[str]:
        return list(self.tables.keys())
    
    def get(self, table: str, sample=False, ctid=False) -> pd.DataFrame:
        db_conn: DatabaseConnection = container.get("DatabaseConnection")
        schema = self.get_scehma(table)
        columns = ",".join([col for col in schema.keys()])
        if sample:
            sample_rate = self.get_sample_rate(table)
            sql = f"select {columns} from {table} tablesample bernoulli({sample_rate * 100})" if not ctid else f"select *, ctid from {table} tablesample bernoulli({sample_rate * 100})"
            data = db_conn.execute(sql)
        else:
            sql = f"select {columns} from {table}" if not ctid else f"select *, ctid from {table}"
            data = db_conn.execute(sql)
        return data
    
    def get_sample_path(self, table: str) -> Optional[str]:
        return self.tables[table].sample_path if table in self.tables else None
    
    def get_sample_rate(self, table: str) -> Optional[str]:
        return self.tables[table].sample_rate if table in self.tables else None
    
    def get_index_path(self, table: str) -> Optional[str]:
        return self.tables[table].index_path if table in self.tables else None
    
    def get_scehma(self, table: str) -> Optional[Dict[str, str]]:
        return self.tables[table].schema if table in self.tables else None
    
    def get_columns(self, table: str) -> Optional[List[str]]:
        return list(self.tables[table].schema.keys()) if table in self.tables else None
    
    def get_rows(self, table: str) -> Optional[int]:
        return self.tables[table].rows if table in self.tables else None
    
    def get_unique(self, table_name: str, col_name: str) -> Optional[Tuple[Union[int, str]]]:
        if (table := self.tables.get(table_name)) is not None:
            return table.uniques.get(col_name)
        return None
    
    def get_minmax(self, table_name: str, col_name: str) -> Optional[Tuple[float]]:
        if (table := self.tables.get(table_name)) is not None:
            return table.minmaxs.get(col_name)
        return None
