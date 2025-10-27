from handler.output_handler import OutputHandler
from handler.latency_wrapper import LatencyWrapper
from handler.handler import Handler

class OutputLatencyWrapper(LatencyWrapper):
    def __init__(self, handler: Handler, latency: str):
        super().__init__(handler, latency)
        self.output_handler = OutputHandler(latency)
        self.name = self.output_handler.name
        self._handler_name = f"{self.__class__.__name__}({handler.__class__.__name__})"
    
    def handle(self) -> None:
        super().handle()
        self.output_handler.handle()
    
    def post_query(self) -> None:
        super().post_query()
        self.output_handler.post_query()