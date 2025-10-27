from typing import Any
from container.initialize_level import initialize_level


@initialize_level(1)
class GenModel:
    def gen(self, table, predicates: list[tuple[Any]], n):
        raise NotImplementedError()