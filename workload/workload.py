from typing import Iterator


class Workload:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path

    def __iter__(self) -> Iterator[str]:
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip().strip(';')
                if line:
                    yield line