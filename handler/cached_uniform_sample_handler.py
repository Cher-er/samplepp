import logging
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any

from handler.handler import Handler
from container.container import container
from dataset.dataset import Dataset

logger = logging.getLogger(__name__)


class CachedUniformSampleHandler(Handler):
    """
    [get]
        predicates
    [register]
        samples
    """
    def __init__(self, sample_rate: float = 0.1) -> None:
        self.sample_rate = sample_rate
        self.caches = {}
    
    def initialize(self) -> None:
        dataset: Dataset = container.get("Dataset")
        seed = container.get("seed")
        for table_name in dataset.table_names:
            df = dataset.get(table_name)
            if table_name in dataset._facts:
                self.caches[table_name] = df.sample(frac=self.sample_rate, random_state=seed)
            else:
                self.caches[table_name] = df

    def handle(self) -> None:
        predicates: List[Tuple[Any]] = container.get("predicates")
        samples: Dict[str, Optional[pd.DataFrame]] = self.filter(self.caches, predicates)
        container.register("samples", samples)
    
    def post_query(self) -> None:
        container.remove("samples")
    
    def filter(self, caches, predicates: List[Tuple[Any]]) -> Dict[str, Optional[pd.DataFrame]]:
        dataset: Dataset = container.get("Dataset")
        samples = {}
        for table, sample in caches.items():
            schema: Dict[str, str] = dataset.get_scehma(table)
            for col, op, val in predicates:
                col = col.lower()
                if col not in schema.keys():
                    continue
                match typ := schema.get(col):
                    case "int":
                        val = int(val)
                    case "str":
                        val = str(val)
                    case "num":
                        val = float(val)
                    case _:
                        logger.error(f"Unsupported type '{typ}' for column '{col}")
                        exit(0)

                match op:
                    case 'EQ':
                        sample = sample[sample[col] == val]
                    case 'GT':
                        sample = sample[sample[col] > val]
                    case 'GTE':
                        sample = sample[sample[col] >= val]
                    case 'LT':
                        sample = sample[sample[col] < val]
                    case 'LTE':
                        sample = sample[sample[col] <= val]
            samples[table] = sample
        return samples