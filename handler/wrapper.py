from handler.handler import Handler


class Wrapper(Handler):
    def __init__(self, handler: Handler):
        self.handler = handler
    
    def initialize(self) -> None:
        self.handler.initialize()
    
    def pre_workload(self) -> None:
        self.handler.pre_workload()
    
    def pre_query(self) -> None:
        self.handler.pre_query()
    
    def handle(self) -> None:
        self.handler.handle()
    
    def post_query(self) -> None:
        self.handler.post_query()
    
    def post_workload(self) -> None:
        self.handler.post_workload()
    
    def finalize(self) -> None:
        self.handler.finalize()