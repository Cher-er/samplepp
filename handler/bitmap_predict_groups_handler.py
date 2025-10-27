import logging
import pickle
import os
import time
import numpy as np
from bitarray import bitarray
from typing import Union

from handler.handler import Handler
from container.container import container
from dataset.dataset import Dataset

logger = logging.getLogger(__name__)


class BitmapPredictGroupsHandler(Handler):
    """
    [get]
        groupby_attr
        predicates
    [register]
        predicted_groups
        group_bitmaps
        num_of_bitmap_ops
    """
    def __init__(self, construct=False, save_dir="bitmaps", num_bins=100):
        self.construct = construct
        self.save_dir = save_dir
        self.num_bins = num_bins

    def initialize(self):
        if self.construct:
            self._consturct()

    def _consturct(self) -> None:
        dataset: Dataset = container.get("Dataset")
        db_name = dataset.db_name
        construct_dataset_start_time = time.perf_counter()
        for table_name in dataset._facts:
            construct_table_start_time = time.perf_counter()
            data = dataset.get(table_name, sample=True, ctid=True)
            rows = data.shape[0]
            schema = dataset.get_scehma(table_name)

            save_table_path = os.path.join(self.save_dir, db_name, f"{table_name}_{dataset.get_sample_rate(table_name)}")
            
            # construct index
            index = {
                "rows": rows,
                "ctid": {i: ctid for i, ctid in enumerate(data['ctid'])}
            }
            if not os.path.exists(index_path := os.path.join(save_table_path, "index.pkl")):
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                with open(index_path, "wb") as f:
                    pickle.dump(index, f)
            

            # consturct bitmap
            for col_name, col_type in schema.items():
                construct_column_start_time = time.perf_counter()
                os.makedirs(path := os.path.join(save_table_path, col_name), exist_ok=True)
                match col_type:
                    case "int":
                        self.consturct_cate_bitmaps(data, rows, table_name, col_name, path)
                    case "str":
                        self.consturct_cate_bitmaps(data, rows, table_name, col_name, path)
                    case "num":
                        self.consturct_num_bitmaps(data, rows, table_name, col_name, path)
                construct_column_end_time = time.perf_counter()
                logger.debug(f"consturct bitmaps for column '{col_name}' elapse {construct_column_end_time - construct_column_start_time}s")
            construct_table_end_time = time.perf_counter()
            logger.debug(f"consturct bitmaps for table '{table_name}' elapse {construct_table_end_time - construct_table_start_time}s")

        construct_dataset_end_time = time.perf_counter()
        logger.debug(f"consturct bitmaps for dataset '{db_name}' elapse {construct_dataset_end_time - construct_dataset_start_time}s")

    def handle(self) -> None:
        dataset: Dataset = container.get("Dataset")
        for table_name in dataset.table_names:
            predicted_groups = set(self.predict_groups(table_name))
            container.register("predicted_groups", predicted_groups)
    
    def post_query(self) -> None:
        container.remove("predicted_groups")
        container.remove("num_of_bitmap_ops")
        container.remove("group_bitmaps")
        container.remove("group_counts")

    def predict_groups(self, table_name: str) -> list[Union[int, str]]:
        predicates = container.get("predicates")
        groupby_attr = container.get("groupby_attr")

        dataset: Dataset = container.get("Dataset")
        db_name = dataset.db_name
        schema = dataset.get_scehma(table_name)

        bitmap_dir = os.path.join(self.save_dir, db_name, f"{table_name}_{dataset.get_sample_rate(table_name)}")

        with open(os.path.join(bitmap_dir, "index.pkl"), 'rb') as f:
            index = pickle.load(f)
            rows = index['rows']

        num_of_bitmap_ops = 0
        bitmap = bitarray(rows)
        bitmap.setall(1)
        locate_bins = {}


        for col_name, op, col_val in predicates:
            
            col_bitmap = None
            match schema.get(col_name):
                case "int":
                    col_val = int(col_val)
                case "str":
                    col_val = col_val
                case "num":
                    col_val = float(col_val)
            
            if op == 'EQ':
                if not os.path.exists(bitmap_path := os.path.join(bitmap_dir, col_name, f"{col_val}.bits")):
                    logger.error(f"bitmap file don't exist: {bitmap_path}")
                else:
                    col_bitmap = bitarray()
                    with open(bitmap_path, 'rb') as f:
                        col_bitmap.fromfile(f)
                        col_bitmap = col_bitmap[:rows]
            elif op in ('GT', 'GTE', 'LT', 'LTE'):
                bins, locate_bin, count_rate = self.get_bins_fully_included(table_name, col_name, op, col_val)
                locate_bins[col_name] = (locate_bin, count_rate)
                col_bitmap = bitarray(rows)
                col_bitmap.setall(0)
                for bin in bins:
                    bin_bitmap = bitarray()
                    if not os.path.exists(bitmap_path := os.path.join(bitmap_dir, col_name, f"{bin}.bits")):
                        logger.error(f"bitmap file don't exist: {bitmap_path}")
                    else:
                        with open(bitmap_path, 'rb') as f:
                            bin_bitmap.fromfile(f)
                            bin_bitmap = bin_bitmap[:rows]
                            col_bitmap |= bin_bitmap
                            num_of_bitmap_ops += 1
            if col_bitmap is not None:
                bitmap &= col_bitmap
                num_of_bitmap_ops += 1
        
        all_groups = dataset.get_unique(table_name, groupby_attr)
        predicted_groups = []
        group_bitmaps = {}
        group_counts = {}
        for group in all_groups:
            if not os.path.exists(bitmap_path := os.path.join(bitmap_dir, groupby_attr, f"{group}.bits")):
                logger.error(f"bitmap file don't exist: {bitmap_path}")
            else:
                group_bitmap = bitarray()
                with open(bitmap_path, 'rb') as f:
                    group_bitmap.fromfile(f)
                    group_bitmap = group_bitmap[:rows]
                    group_bitmap &= bitmap
                    num_of_bitmap_ops += 1
                    if group_bitmap.any():
                        base_count = group_bitmap.count()
                        predicted_groups.append(group)
                        group_bitmaps[group] = group_bitmap
                        # deal locate_bin
                        extra_count = 0
                        for col_name, (locate_bin, count_rate) in locate_bins.items():
                            if not os.path.exists(bitmap_path := os.path.join(bitmap_dir, col_name, f"{locate_bin}.bits")):
                                logger.error(f"bitmap file don't exist: {bitmap_path}")
                            else:
                                with open(bitmap_path, 'rb') as f:
                                    locate_bin_bitmap = bitarray()
                                    locate_bin_bitmap.fromfile(f)
                                    locate_bin_bitmap = locate_bin_bitmap[:rows]
                                    locate_bin_bitmap &= group_bitmap
                                    num_of_bitmap_ops += 1
                                    extra_count += locate_bin_bitmap.count() * count_rate
                        group_counts[group] = (base_count + extra_count) / dataset.get_sample_rate(table_name)
        container.register("num_of_bitmap_ops", num_of_bitmap_ops)
        container.register("group_bitmaps", group_bitmaps)
        container.register("group_counts", group_counts)
        return predicted_groups
    
    def get_bins_fully_included(self, table_name, col_name, op, col_val) -> list[int]:
        dataset: Dataset = container.get("Dataset")
        col_min, col_max = dataset.get_minmax(table_name, col_name)

        bin_len = (col_max - col_min) / self.num_bins
        locate_bin = int((col_val - col_min) / bin_len)
        locate_bin_left = col_min + bin_len * locate_bin
        locate_bin_right = col_min + bin_len * (locate_bin + 1)

        if op in ("GT", "GTE"):
            count_rate = (locate_bin_right - col_val) / bin_len
            return list(range(locate_bin + 1, self.num_bins)), locate_bin, count_rate
        if op in ("LT", "LTE"):
            count_rate = (col_val - locate_bin_left) / bin_len
            return list(range(0, locate_bin)), locate_bin, count_rate

    def consturct_cate_bitmaps(self, data, rows, table_name, col_name, path):
        dataset: Dataset = container.get("Dataset")
        for col_val in dataset.get_unique(table_name, col_name):
            if not os.path.exists(bitmap_path := os.path.join(path, f"{col_val}.bits")):
                bitmap = bitarray(rows)
                bitmap.setall(0)
                mask = data[col_name] == col_val
                for i, is_match in enumerate(mask):
                    if is_match:
                        bitmap[i] = 1
                with open(bitmap_path, 'wb') as f:
                    bitmap.tofile(f)
    
    def consturct_num_bitmaps(self, data, rows, table_name, col_name, path):
        dataset: Dataset = container.get("Dataset")
        col_min, col_max = dataset.get_minmax(table_name, col_name)
        edges = np.linspace(col_min, col_max, self.num_bins + 1)
        for i in range(self.num_bins):
            if not os.path.exists(bitmap_path := os.path.join(path, f"{i}.bits")):
                bitmap = bitarray(rows)
                bitmap.setall(0)
                left = edges[i]
                right = edges[i + 1]
                mask = (data[col_name] >= left) & (data[col_name] <= right)
                for i, is_match in enumerate(mask):
                    if is_match:
                        bitmap[i] = 1
                with open(bitmap_path, 'wb') as f:
                    bitmap.tofile(f)
