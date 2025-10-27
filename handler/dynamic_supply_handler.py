import random
import pandas as pd
from typing import Optional

from handler.handler import Handler
from container.container import container
from gen_model.gen_model import GenModel
from dataset.dataset import Dataset


class DynamicSupplyHandler(Handler):
    """
    [get]
        samples
        aggregate_attr
        groupby_attr
        predicates
    [register or update]
        generatives
    """
    def __init__(self, gen_size=10, max_iter=10, epsilon=0.5):
        self.gen_size = gen_size
        self.max_iter = max_iter
        self.epsilon = epsilon

    def handle(self) -> None:
        samples: Optional[dict[str, pd.DataFrame]] = container.get("samples")
        generatives: Optional[dict[str, pd.DataFrame]] = container.get("generatives", {})
        aggregate_attr = container.get("aggregate_attr")
        groupby_attr = container.get("groupby_attr")
        predicates = container.get("predicates")
        gen_model: GenModel = container.get("GenModel")
        dataset: Dataset = container.get("Dataset")

        for table_name, sample in samples.items():
            if table_name not in dataset._facts:
                continue
            if sample.shape[0] == 0:
                return
            
            schema = dataset.get_scehma(table_name)
            table_predicates = []
            for predicate in predicates:
                if predicate[0] in schema.keys():
                    table_predicates.append(predicate)
            
            groups_counts: dict = sample.groupby(groupby_attr).size().to_dict()
            groups = list(groups_counts.keys())

            all_generative = {}
            rewards = {}
            for _ in range(self.max_iter):
                if random.random() < self.epsilon or not rewards:
                    # explore
                    counts = list(groups_counts.values())
                    weights = [self.weight_function(count) for count in counts]
                    total_weight = sum(weights)
                    if total_weight == 0:
                        break
                    probabilities = [w / total_weight for w in weights]
                    group = random.choices(groups, weights=probabilities, k=1)[0]
                else:
                    # exploit
                    group = max(rewards, key=rewards.get)
                
                generative = gen_model.gen(table_name, table_predicates + [(groupby_attr, 'EQ', group)], n=self.gen_size)

                if generative is None:
                    continue

                if_not_accept = pd.concat([sample, all_generative[group]], ignore_index=True) if group in all_generative else sample
                if_accept = pd.concat([if_not_accept, generative], ignore_index=True)
                if_not_accept_cv = if_not_accept[aggregate_attr].std() / if_not_accept[aggregate_attr].mean()
                if_accept_cv = if_accept[aggregate_attr].std() / if_accept[aggregate_attr].mean()

                if if_not_accept_cv > if_accept_cv:
                    all_generative[group] = generative if group not in all_generative else pd.concat([all_generative[group], generative], ignore_index=True)
                    rewards[group] = if_not_accept_cv - if_accept_cv
                    groups_counts[group] += self.gen_size
                else:
                    groups_counts[group] = float("inf")
                    if group in rewards:
                        del rewards[group]
            
            if all_generative:
                generatives[table_name] = pd.concat([origin, *[generative for generative in all_generative.values()]] if ((origin := generatives.get(table_name)) is not None) else all_generative, ignore_index=True)
        
        if generatives:
            container.register_or_update("generatives", generatives)
    
    def post_query(self) -> None:
        container.try_remove("generatives")
    
    def weight_function(self, count):
        return 1 / (1 + count / 100)
        