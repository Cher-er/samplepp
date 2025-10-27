import os
import logging
import pandas as pd
from typing import Any

from gen_model.gen_model import GenModel
from gen_model.vaeac.generator import Generator
from dataset.dataset import Dataset
from container.container import container

logger = logging.getLogger(__name__)


class VaeacModel(GenModel):
    def __init__(self, epochs=10) -> None:
        self.best_state_paths = {}
        self.models: dict[str, Generator] = {}
        self.epochs: int = epochs

    def initialize(self) -> None:
        dataset: Dataset = container.get("Dataset")
        for table_name in dataset._facts:
             self.best_state_paths[table_name] = dataset.best_state_paths.get(table_name)
             self.models[table_name] = self._train_or_load(table_name, dataset)
    
    def _train_or_load(self, table_name: str, dataset: Dataset) -> Generator:
        data = dataset.get(table_name)
        raw_schema = dataset.get_scehma(table_name)
        schema = {col: ('n' if typ == 'num' else 'c') for col, typ in raw_schema.items()}

        if os.path.exists(path := self.best_state_paths.get(table_name)):
            logger.info(f"Load best_state from {path}")
            return Generator(data, schema=schema, load_state=path)
        else:
            logger.info(f"Training model for table '{table_name}'")
            generator = Generator(data, schema=schema)
            generator.train(self.epochs, mask="ran 0.5", save_file=path)
            logger.info(f"Train model to {path}")
            return generator
    
    def gen(self, table: str, predicates: list[tuple[Any]], n) -> pd.DataFrame:
        dataset: Dataset = container.get("Dataset")
        schema = dataset.get_scehma(table)
        generator = self.models[table]

        predicates_filter = []
        for predicate in predicates:
            if predicate[0] in schema:
                predicates_filter.append(predicate)
        predicates = predicates_filter

        try:
            conditions = [self._parse_predicate(predicate, schema) for predicate in predicates]
        except ValueError as e:
            logger.warning(str(e))
            return None

        return generator.gen(conditions, n, False)
    
    def _parse_predicate(self, predicate: tuple[Any], schema: dict[str, str]) -> tuple[str, str, Any]:
        col, op, val = predicate
        typ = schema.get(col)
        match typ:
            case "int":
                val = int(val)
            case "str":
                val = str(val)
            case "num":
                val = float(val)
            case _:
                raise ValueError(f"Unsupported column type '{typ}' for column '{col}'")
        
        op_map = {
            'EQ': '=',
            'GT': '>',
            'GTE': '>=',
            'LT': '<',
            'LTE': '<='
        }

        if (mapped_op := op_map.get(op)) is None:
            raise ValueError(f"Invalid operatoe: {op}")
        
        return col, mapped_op, val
