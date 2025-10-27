import os
import csv

from handler.handler import Handler
from container.container import container
from pipeline.pipeline import Pipeline
from workload.workload import Workload
from handler.output_handler import OutputHandler
from handler.output_latency_wrapper import OutputLatencyWrapper


class SaveOutputHandler(Handler):
    """
    [get]
        output
    """
    def __init__(self, output_dir: str="out") -> None:
        self.output_dir = output_dir
        self.headers: list[str] = []
        self.file = None
        self.writer = None
        
    def pre_workload(self) -> None:
        pipeline: Pipeline = container.get("Pipeline")
        workload: Workload = container.get("Workload")
        args = container.get("args")

        self.headers = [
            handler.name
            for handler in pipeline
            if isinstance(handler, OutputHandler) or isinstance(handler, OutputLatencyWrapper)
        ]
        
        os.makedirs((output_dir := os.path.join(self.output_dir, args.dataset, workload.name)), exist_ok=True)
        file_path  = os.path.join(output_dir, f"output.csv")
        self.file = open(file_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=self.headers)
        self.writer.writeheader()
    
    def handle(self) -> None:
        output = container.get("output")
        if output:
            self.writer.writerow(self._format_output(output))
    
    def post_workload(self) -> None:
        if self.file:
            self.file.close()
            self.file = None
    
    def _format_output(self, row: dict, float_fmt: str = "{:.4f}") -> dict:
        return {
            k: (float_fmt.format(v) if isinstance(v, float) else v)
            for k, v in row.items()
            if k in self.headers
        }