import logging
import warnings
import os
import random
import argparse
import numpy as np
import torch

from container.container import container
from engine.engine import Engine
from pipeline.pipeline import Pipeline
from gen_model.vaeac_model import VaeacModel
from database_connection.database_connection import DatabaseConnection

from handler.output_latency_wrapper import OutputLatencyWrapper
from handler.output_handler import OutputHandler
from handler.save_output_handler import SaveOutputHandler
from handler.existed_ground_truth_handler import ExistedGroundTruthHandler
from handler.query_info_handler import QueryInfoHandler
from handler.ground_truth_handler import GroundTruthHandler
from handler.ground_truth_sample_handler import GroundTruthSampleHandler
from handler.supply_missing_groups_handler import SupplyMissingGroupsHandler
from handler.cached_uniform_sample_handler import CachedUniformSampleHandler
from handler.avg_approximate_query_processing_handler import AvgApproximateQueryProcessingHandler
from handler.count_approximate_query_processing_handler import CountApproximateQueryProcessingHandler
from handler.sum_approximate_query_processing_handler import SumApproximateQueryProcessingHandler
from handler.relative_error_handler import RelativeErrorHandler
from handler.bitmap_predict_groups_handler import BitmapPredictGroupsHandler
from handler.roaring_bitmap_predict_groups_handler import RoaringBitmapPredictGroupsHandler
from handler.bitmap_dynamic_sampling_handler import BitmapDynamicSamplingHandler
from handler.roaring_bitmap_dynamic_sampling_handler import RoaringBitmapDynamicSamplingHandler
from handler.dynamic_supply_handler import DynamicSupplyHandler
from handler.save_ground_truth_handler import SaveGroundTruthHandler

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _init_seed(seed=42):
    container.register("seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init():
    _init_seed()
    _init_args()
    _init_container()


def _init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload_names', nargs='+', required=True)
    parser.add_argument('--workload_paths', nargs='+', required=True)
    parser.add_argument('--agg', type=str, required=True)
    parser.add_argument('--ground_truth_paths', nargs='+', default=[])
    parser.add_argument('--output_dir', type=str, default="out")
    parser.add_argument('--dataset', type=str, default="flights")
    parser.add_argument('--sample_rate', type=float, default=0.1)
    parser.add_argument('--sampling_size', type=int, default=100)
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--gen_missing', action='store_true')
    parser.add_argument('--ground_truth_sample', action='store_true')
    parser.add_argument('--roaring_bitmap', action='store_true')
    args = parser.parse_args()
    container.register("args", args)


def _init_container():
    args = container.get("args")
    match args.dataset:
        case "flights":
            from dataset.flights import Flights
            container.register("Dataset", Flights())
        case "flights_20m":
            from dataset.flights_20m import Flights20M
            container.register("Dataset", Flights20M())
        case "flights_40m":
            from dataset.flights_40m import Flights40M
            container.register("Dataset", Flights40M())
        case "flights_100m":
            from dataset.flights_100m import Flights100M
            container.register("Dataset", Flights100M())
        case "tpcds_1g":
            from dataset.tpcds_1g import Tpcds1g
            container.register("Dataset", Tpcds1g())
        case "tpcds_50g":
            from dataset.tpcds_50g import Tpcds50g
            container.register("Dataset", Tpcds50g())
        case _:
            logger.error(f"Unknown dataset: {args.dataset}")
            exit(0)

    container.register("Engine", Engine())
    container.register("GenModel", VaeacModel())
    container.register("DatabaseConnection", DatabaseConnection())

    match args.agg:
        case "avg":
            container.register("Pipeline", construct_pipeline_avg())
        case "count":
            container.register("Pipeline", construct_pipeline_count())
        case "sum":
            container.register("Pipeline", construct_pipeline_sum())
        case "avg_vaeac":
            container.register("Pipeline", construct_pipeline_avg_vaeac())
        case _:
            logger.error(f"Unsupported aggregate function: {args.agg}")
            exit(0)


def construct_pipeline_avg():
    args = container.get("args")
    pipeline = Pipeline()

    pipeline.append(OutputHandler("idx"))

    # ground truth
    if args.ground_truth_paths:
        pipeline.append(ExistedGroundTruthHandler(args.ground_truth_paths))
    else:
        pipeline.append(OutputLatencyWrapper(GroundTruthHandler(), "ground_truth_latency"))
    
    # ground truth sample
    if args.ground_truth_sample:
        pipeline.append(GroundTruthSampleHandler())

    pipeline.append(QueryInfoHandler())

    # samples
    pipeline.append(OutputLatencyWrapper(CachedUniformSampleHandler(args.sample_rate), "samples_latency"))

    # aqp
    pipeline.append(AvgApproximateQueryProcessingHandler("result_with_samples"))
    pipeline.append(RelativeErrorHandler("re_with_samples", "result_with_samples"))
    pipeline.append(OutputHandler("re_with_samples"))

    # predict groups
    if args.roaring_bitmap:
        pipeline.append(OutputLatencyWrapper(RoaringBitmapPredictGroupsHandler(construct=True), "predict_groups_latency"))
    else:
        pipeline.append(OutputLatencyWrapper(BitmapPredictGroupsHandler(construct=True), "predict_groups_latency"))
    pipeline.append(OutputHandler("num_of_bitmap_ops"))


    if args.gen_missing:
        # supply missing groups
        pipeline.append(OutputLatencyWrapper(SupplyMissingGroupsHandler(), "supply_missing_groups_latency"))
        pipeline.append(OutputHandler("num_of_missing_groups"))
    
        # aqp
        pipeline.append(AvgApproximateQueryProcessingHandler("result_after_supply_missing_groups"))
        pipeline.append(RelativeErrorHandler("re_after_supply_missing_groups", "result_after_supply_missing_groups"))
        pipeline.append(OutputHandler("re_after_supply_missing_groups"))
    else:
        # sampling rare groups
        if args.roaring_bitmap:
            pipeline.append(OutputLatencyWrapper(RoaringBitmapDynamicSamplingHandler(sampling_size=args.sampling_size), "sampling_rare_groups_latency"))
        else:
            pipeline.append(OutputLatencyWrapper(BitmapDynamicSamplingHandler(sampling_size=args.sampling_size), "sampling_rare_groups_latency"))
        pipeline.append(OutputHandler("num_of_rare_groups"))
    
        # aqp
        pipeline.append(AvgApproximateQueryProcessingHandler("result_after_sampling_rare_groups"))
        pipeline.append(RelativeErrorHandler("re_after_sampling_rare_groups", "result_after_sampling_rare_groups"))
        pipeline.append(OutputHandler("re_after_sampling_rare_groups"))


    # dynamic supply
    pipeline.append(OutputLatencyWrapper(DynamicSupplyHandler(max_iter=args.max_iter), "dynamic_supply_latency"))

    # aqp
    pipeline.append(AvgApproximateQueryProcessingHandler("result_after_dynamic_supply"))
    pipeline.append(RelativeErrorHandler("re_after_dynamic_supply", "result_after_dynamic_supply"))
    pipeline.append(OutputHandler("re_after_dynamic_supply"))
    pipeline.append(OutputHandler("precision"))
    pipeline.append(OutputHandler("recall"))
    pipeline.append(OutputHandler("missing_rate"))
    pipeline.append(OutputHandler("out_rate"))
    pipeline.append(OutputHandler("process_latency"))

    # save output
    pipeline.append(SaveOutputHandler(args.output_dir))
    if not args.ground_truth_paths:
        pipeline.append(SaveGroundTruthHandler())

    return pipeline


def construct_pipeline_avg_vaeac():
    args = container.get("args")
    pipeline = Pipeline()

    pipeline.append(OutputHandler("idx"))

    # ground truth
    if args.ground_truth_paths:
        pipeline.append(ExistedGroundTruthHandler(args.ground_truth_paths))
    else:
        pipeline.append(OutputLatencyWrapper(GroundTruthHandler(), "ground_truth_latency"))

    pipeline.append(QueryInfoHandler())

    # dynamic supply
    pipeline.append(OutputLatencyWrapper(DynamicSupplyHandler(max_iter=args.max_iter), "dynamic_supply_latency"))

    # aqp
    pipeline.append(AvgApproximateQueryProcessingHandler("result_after_dynamic_supply"))
    pipeline.append(RelativeErrorHandler("re_after_dynamic_supply", "result_after_dynamic_supply"))
    pipeline.append(OutputHandler("re_after_dynamic_supply"))

    # save output
    pipeline.append(SaveOutputHandler(args.output_dir))
    if not args.ground_truth_paths:
        pipeline.append(SaveGroundTruthHandler())

    return pipeline


def construct_pipeline_count():
    args = container.get("args")
    pipeline = Pipeline()
    pipeline.append(OutputHandler("idx"))

    # ground truth
    if args.ground_truth_paths:
        pipeline.append(ExistedGroundTruthHandler(args.ground_truth_paths))
    else:
        pipeline.append(OutputLatencyWrapper(GroundTruthHandler(), "ground_truth_latency"))
    
    pipeline.append(QueryInfoHandler())

    # samples
    pipeline.append(OutputLatencyWrapper(CachedUniformSampleHandler(args.sample_rate), "samples_latency"))

    # predict groups
    if args.roaring_bitmap:
        pipeline.append(OutputLatencyWrapper(RoaringBitmapPredictGroupsHandler(construct=True), "predict_groups_latency"))
    else:
        pipeline.append(OutputLatencyWrapper(BitmapPredictGroupsHandler(construct=True), "predict_groups_latency"))
    pipeline.append(OutputHandler("num_of_bitmap_ops"))

    # aqp
    pipeline.append(CountApproximateQueryProcessingHandler("result"))
    pipeline.append(RelativeErrorHandler("re", "result"))
    pipeline.append(OutputHandler("re"))
    pipeline.append(OutputHandler("precision"))
    pipeline.append(OutputHandler("recall"))
    pipeline.append(OutputHandler("missing_rate"))
    pipeline.append(OutputHandler("out_rate"))
    pipeline.append(OutputHandler("process_latency"))

    # save
    pipeline.append(SaveOutputHandler(args.output_dir))
    if not args.ground_truth_paths:
        pipeline.append(SaveGroundTruthHandler())

    return pipeline


def construct_pipeline_sum():
    args = container.get("args")
    pipeline = Pipeline()

    pipeline.append(OutputHandler("idx"))

    # ground truth
    if args.ground_truth_paths:
        pipeline.append(ExistedGroundTruthHandler(args.ground_truth_paths))
    else:
        pipeline.append(OutputLatencyWrapper(GroundTruthHandler(), "ground_truth_latency"))
    
    # ground truth sample
    if args.ground_truth_sample:
        pipeline.append(GroundTruthSampleHandler())

    pipeline.append(QueryInfoHandler())

    # samples
    pipeline.append(OutputLatencyWrapper(CachedUniformSampleHandler(args.sample_rate), "samples_latency"))

    # predict groups
    if args.roaring_bitmap:
        pipeline.append(OutputLatencyWrapper(RoaringBitmapPredictGroupsHandler(construct=True), "predict_groups_latency"))
    else:
        pipeline.append(OutputLatencyWrapper(BitmapPredictGroupsHandler(construct=True), "predict_groups_latency"))
    pipeline.append(OutputHandler("num_of_bitmap_ops"))


    if args.gen_missing:
        # supply missing groups
        pipeline.append(OutputLatencyWrapper(SupplyMissingGroupsHandler(), "supply_missing_groups_latency"))
        pipeline.append(OutputHandler("num_of_missing_groups"))
    
        # aqp
        pipeline.append(SumApproximateQueryProcessingHandler("result_after_supply_missing_groups"))
        pipeline.append(RelativeErrorHandler("re_after_supply_missing_groups", "result_after_supply_missing_groups"))
        pipeline.append(OutputHandler("re_after_supply_missing_groups"))
    else:
        # sampling rare groups
        if args.roaring_bitmap:
            pipeline.append(OutputLatencyWrapper(RoaringBitmapDynamicSamplingHandler(sampling_size=args.sampling_size), "sampling_rare_groups_latency"))
        else:
            pipeline.append(OutputLatencyWrapper(BitmapDynamicSamplingHandler(sampling_size=args.sampling_size), "sampling_rare_groups_latency"))
        pipeline.append(OutputHandler("num_of_rare_groups"))
    
        # aqp
        pipeline.append(SumApproximateQueryProcessingHandler("result_after_sampling_rare_groups"))
        pipeline.append(RelativeErrorHandler("re_after_sampling_rare_groups", "result_after_sampling_rare_groups"))
        pipeline.append(OutputHandler("re_after_sampling_rare_groups"))

    # dynamic supply
    pipeline.append(OutputLatencyWrapper(DynamicSupplyHandler(max_iter=args.max_iter), "dynamic_supply_latency"))

    # aqp
    pipeline.append(SumApproximateQueryProcessingHandler("result_after_dynamic_supply"))
    pipeline.append(RelativeErrorHandler("re_after_dynamic_supply", "result_after_dynamic_supply"))
    pipeline.append(OutputHandler("re_after_dynamic_supply"))
    pipeline.append(OutputHandler("precision"))
    pipeline.append(OutputHandler("recall"))
    pipeline.append(OutputHandler("missing_rate"))
    pipeline.append(OutputHandler("out_rate"))
    pipeline.append(OutputHandler("process_latency"))

    # save output
    pipeline.append(SaveOutputHandler(args.output_dir))
    if not args.ground_truth_paths:
        pipeline.append(SaveGroundTruthHandler())

    return pipeline


def main():
    engine: Engine = container.get("Engine")
    engine.start()


def clear():
    container.remove("args")
    container.remove("Dataset")
    container.remove("Engine")
    container.remove("GenModel")
    container.remove("Pipeline")


if __name__ == '__main__':
    init()
    main()
    clear()