from handler.handler import Handler
from container.container import container


class OutputHandler(Handler):
    """
    [register or update]
        output
    """
    def __init__(self, name: str, target: str = None) -> None:
        self.name = name
        self.target = target if target else name
    
    def handle(self) -> None:
        target = container.get(self.target)
        output = container.get("output", {})
        output[self.name] = target
        container.register_or_update("output", output)
    
    def post_query(self) -> None:
        container.try_remove("output")
    
