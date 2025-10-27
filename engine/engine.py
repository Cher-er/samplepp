import logging

from container.container import container
from workload.workload import Workload
from query.query import Query
from pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


class Engine:
    def start(self) -> None:
        self._initialize_components()

        args = container.get("args")
        for name, path in zip(args.workload_names, args.workload_paths):
            self._process_workload(name, path)

        self._finalize_components()

    def _initialize_components(self) -> None:
        items = sorted(
            container.items(),
            key=lambda item: getattr(item[1], "__initialize_level", 0)
        )
        for name, obj in items:
            if hasattr(obj, "initialize"):
                logger.debug(f"Initializing [{getattr(obj, '__initialize_level', 0)}]: {name}")
                obj.initialize()

    def _finalize_components(self) -> None:
        for name, obj in container.items():
            if hasattr(obj, "finalize"):
                logger.debug(f"Finalizing: {name}")
                obj.finalize()
    
    def _process_workload(self, name: str, path: str) -> None:
        logger.info(f"==== Workload: {name} ====")

        workload = Workload(name=name, path=path)
        container.register("Workload", workload)

        pipeline: Pipeline = container.get("Pipeline")
        pipeline.pre_workload()

        for idx, sql in enumerate(workload):
            self._process_query(idx, sql, pipeline)

        pipeline.post_workload()
        container.remove("Workload")
    
    def _process_query(self, idx: int, sql: str, pipeline: Pipeline) -> None:
        logger.info(f"SQL {idx}: {sql}")

        query = Query(sql)
        container.register("idx", idx)
        container.register("query", query)

        try:
            pipeline.process()
        except Exception as e:
            logger.exception(f"Error in SQL {idx}: {sql}")
        finally:
            container.try_remove("idx")
            container.try_remove("query")