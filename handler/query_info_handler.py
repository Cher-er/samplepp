from typing import Any

from handler.handler import Handler
from container.container import container
from query.query import Query
from dataset.dataset import Dataset


class QueryInfoHandler(Handler):
    """
    [register]
        aggregate_func
        aggregate_attr
        from_table
        predicates
        groupby_attr
        groups
    """
    def handle(self) -> None:
        query: Query = container.get("query")
        groupby_attr = query.get_groupby_attrs()[0]
        from_table = query.get_from_table()
        join_table = query.get_join_table()
        join_attrs = query.get_join_attrs()
        predicates = query.get_predicates()
        aggregate_attr = query.get_aggregate_attr()
        aggregate_func = query.get_aggregate_func()
        container.register("groupby_attr", groupby_attr)
        container.register("from_table", from_table)
        container.register("join_table", join_table)
        container.register("join_attrs", join_attrs)
        container.register("predicates", predicates)
        container.register("aggregate_attr", aggregate_attr)
        container.register("aggregate_func", aggregate_func)

        dataset: Dataset = container.get("Dataset")
        groups: list[Any] = dataset.get_unique(from_table, groupby_attr)
        container.register("groups", groups)
    
    def post_query(self) -> None:
        container.remove("groupby_attr")
        container.remove("from_table")
        container.remove("join_table")
        container.remove("join_attrs")
        container.remove("predicates")
        container.remove("aggregate_attr")
        container.remove("aggregate_func")
        container.remove("groups")