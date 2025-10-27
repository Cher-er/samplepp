class Table:
    def __init__(self, schema: dict[str, str], rows: int, sample_rate: float = 1.0) -> None:
        self.schema = schema
        self.rows = rows
        self.sample_rate = sample_rate
        self.uniques = {}
        self.minmaxs = {}
       