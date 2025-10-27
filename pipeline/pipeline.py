import logging
from typing import Iterable

from handler.handler import Handler
from container.initialize_level import initialize_level

logger = logging.getLogger(__name__)


@initialize_level(2)
class Pipeline:
    def __init__(self) -> None:
        self.pipeline: list[Handler] = []
    
    def append(self, handler: Handler) -> None:
        self.pipeline.append(handler)
    
    def process(self) -> None:
        self.pre_query()
        self.handle()
        self.post_query()
    
    def initialize(self) -> None:    
        hanlders = sorted(
            self.pipeline,
            key=lambda handler: getattr(handler, "__initialize_level", 0)
        )
        for handler in hanlders:
            logger.debug(f"initialize handler {getattr(handler, '_handler_name', handler.__class__.__name__)}")
            handler.initialize()
    
    def pre_workload(self) -> None:
        for handler in reversed(self.pipeline):
            handler.pre_workload()
    
    def pre_query(self) -> None:
        for handler in self.pipeline:
            handler.pre_query()

    def handle(self) -> None:
        for handler in self.pipeline:
            handler.handle()
    
    def post_query(self) -> None:
        for handler in reversed(self.pipeline):
            handler.post_query()

    def post_workload(self) -> None:
        for handler in reversed(self.pipeline):
            handler.post_workload()
    
    def finalize(self) -> None:
        for handler in self.pipeline:
            logger.debug(f"finalize handler {handler.__class__.__name__}")
            handler.finalize()
    
    def __iter__(self) -> Iterable[Handler]:
        return iter(self.pipeline)