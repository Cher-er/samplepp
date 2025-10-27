import logging
import pandas as pd

from handler.handler import Handler
from container.container import container
from gen_model.gen_model import GenModel

logger = logging.getLogger(__name__)


class SupplyMissingGroupsHandler(Handler):
    """
    [get]
        samples
        groupby_attr
        predicates
        predicted_groups
    [register]
        generatives
        num_of_missing_groups
    """
    def __init__(self, gen_size=10):
        self.gen_size = gen_size

    def handle(self) -> None:
        samples: dict[str, pd.DataFrame] = container.get("samples")
        generatives: dict[str, pd.DataFrame] = container.get("generatives", {})

        groupby_attr = container.get("groupby_attr")
        gen_model: GenModel = container.get("GenModel")
        predicates = container.get("predicates")
        predicted_groups = container.get("predicted_groups")
        
        for table_name, sample in samples.items():
            if groupby_attr not in sample.columns:
                continue
            existed_groups = set(sample[groupby_attr].unique())
            missing_groups = predicted_groups - existed_groups
            container.register("num_of_missing_groups", len(missing_groups))

            all_generative = []
            for missing_group in missing_groups:
                generative = gen_model.gen(table_name, predicates + [(groupby_attr, 'EQ', missing_group)], n=self.gen_size)
                all_generative.append(generative)
            if all_generative:
                generatives[table_name] = pd.concat(all_generative, ignore_index=True)
        
        if generatives:
            container.register("generatives", generatives)
    
    def post_query(self) -> None:
        container.try_remove("generatives")
        container.remove("num_of_missing_groups")