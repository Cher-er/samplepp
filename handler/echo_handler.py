import logging

from handler.handler import Handler
from container.container import container

logger = logging.getLogger(__name__)


class EchoHandler(Handler):
    def __init__(self, name: str, target: str = None) -> None:
        self.name = name
        self.target = target if target else name
    
    def handle(self) -> None:
        target = container.get(self.target)
        logger.info(f"[Echo] {self.name}:\n{target}")