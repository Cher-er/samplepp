import time

from container.container import container
from handler.handler import Handler
from handler.wrapper import Wrapper


class LatencyWrapper(Wrapper):
    def __init__(self, handler: Handler, latency: str):
        super().__init__(handler)
        self.latency = latency
    
    def handle(self) -> None:
        start_time = time.perf_counter()
        self.handler.handle()
        end_time = time.perf_counter()
        container.register(self.latency, end_time - start_time)
    
    def post_query(self) -> None:
        self.handler.post_query()
        container.remove(self.latency)