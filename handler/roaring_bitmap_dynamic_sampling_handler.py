import logging
import os
import pickle
import random
import pandas as pd
from pyroaring import BitMap

from handler.handler import Handler
from container.container import container
from container.initialize_level import initialize_level
from dataset.dataset import Dataset
from database_connection.database_connection import DatabaseConnection

logger = logging.getLogger(__name__)


@initialize_level(1)
class RoaringBitmapDynamicSamplingHandler(Handler):
    """
    [get]
        predicted_groups
        group_bitmaps
        groupby_attr
        samples
    """
    def __init__(self, sampling_size=100, load_dir="roaring_bitmaps") -> None:
        self.sampling_size = sampling_size
        self.load_dir = load_dir
        self.indices: dict[str, dict] = {}

    def initialize(self) -> None:
        dataset: Dataset = container.get("Dataset")
        db_name = dataset.db_name

        for table_name in dataset._facts:
            if not os.path.exists(path := os.path.join(self.load_dir, db_name, f"{table_name}_{dataset.get_sample_rate(table_name)}", "index.pkl")):
                logger.error(f"index file don't exist: {path}")
                exit(0)
            else:
                with open(path, 'rb') as f:
                    index = pickle.load(f)['ctid']  # row -> ctid
                    self.indices[table_name] = index
    
    def handle(self) -> None:
        predicted_groups = container.get("predicted_groups")
        group_bitmaps = container.get("group_bitmaps")
        groupby_attr = container.get("groupby_attr")
        dataset: Dataset = container.get("Dataset")
        db_conn: DatabaseConnection = container.get("DatabaseConnection")
        samples: dict[str, pd.DataFrame] = container.get("samples")

        for table_name in dataset._facts:
            sample = samples.get(table_name)
            groups_counts: dict = sample.groupby(groupby_attr).size().to_dict()

            data = []
            num_of_rare_groups = 0
            for group in predicted_groups:
                count = groups_counts.get(group, 0)
                if count >= self.sampling_size:
                    continue
                group_bitmap: BitMap = group_bitmaps.get(group)
                if not group_bitmap:
                    continue
                num_of_rare_groups += 1
                rows = list(group_bitmap)
                rows = random.sample(rows, min(self.sampling_size - count, len(rows)))

                sql = f"select * from {table_name} where ctid in ("
                for row in rows:
                    ctid = self.indices[table_name][row]
                    sql += f"'{ctid}',"
                sql = sql[:-1] + ")"

                sampling_data = db_conn.execute(sql)
                data.append(sampling_data)
            if data:
                data = pd.concat(data, ignore_index=True)
                data = data.astype(sample.dtypes.to_dict())
                samples[table_name] = pd.concat([sample, data], ignore_index=True)
        
        container.update("samples", samples)
        container.register("num_of_rare_groups", num_of_rare_groups)
    
    def post_query(self) -> None:
        container.remove("num_of_rare_groups")