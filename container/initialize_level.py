def initialize_level(level: int):
    """
    the larger the level, the later the initialization
    """
    def decorator(cls):
        cls.__initialize_level = level
        return cls
    return decorator