import logging
from typing import ItemsView, Any

logger = logging.getLogger(__name__)


class Container:
    def __init__(self) -> None:
        self._objs: dict[str, Any] = {}
    
    def get(self, name: str, default: Any = None) -> Any:
        return self._objs.get(name, default)
    
    def register(self, name: str, obj: Any) -> bool:
        return self._safe_set(name, obj, allow_override=False)
    
    def update(self, name: str, obj: Any) -> bool:
        return self._safe_set(name, obj, allow_override=True, must_exist=True)
    
    def register_or_update(self, name: str, obj: Any) -> None:
        self._objs[name] = obj
    
    def remove(self, name: str) -> bool:
        if name in self._objs:
            del self._objs[name]
            return True
        self._log_error(name, "remove", "not found")
        return False
    
    def try_remove(self, name: str) -> None:
        self._objs.pop(name, None)
    
    def exist(self, name: str) -> bool:
        return name in self._objs
    
    def items(self) -> ItemsView[str, Any]:
        return self._objs.items()
    
    def _safe_set(
            self, name: str, obj: Any, *, allow_override: bool = False, must_exist: bool = False
    ) -> bool:
        exists = name in self._objs
        if must_exist and not exists:
            self._log_error(name, "update", "not found")
            return False
        if not allow_override and exists:
            self._log_error(name, "register", "already exists")
            return False
        self._objs[name] = obj
        return True
    
    def _log_error(self, name: str, action: str, reason: str) -> None:
        logger.error(f"Failed to {action} '{name}': {reason}")


container = Container()