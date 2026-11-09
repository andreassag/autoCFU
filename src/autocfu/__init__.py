"""AutoCFU: High-resolution sliced microbial colony detection engine."""

from __future__ import annotations

from .config import EngineConfig, InferenceConfig, SlicingConfig, load_config
from .merger import NMS, Detection, GreedyNMM, calculate_ios, calculate_iou, get_merger
from .model import RTDetrONNXModel, RTDetrV2ONNXModel, RTDetrV4ONNXModel
from .pipeline import AgarInferenceEngine, AutoCFUEngine, InferenceResult, export_batch_summary_csv
from .slicer import PlateSlicer, SliceWindow
from .utils import (
    IGNORED_CATEGORIES,
    KNOWN_COLONY_LABELS,
    compute_detection_metrics,
    extract_gt_colony_boxes,
    is_colony_label,
    is_valid_highres_plate,
    plot_plate_detections,
    stream_sample_plate,
)

__version__ = "0.1.1"

__all__ = [
    "AutoCFUEngine",
    "AgarInferenceEngine",
    "InferenceResult",
    "PlateSlicer",
    "SliceWindow",
    "GreedyNMM",
    "NMS",
    "Detection",
    "calculate_iou",
    "calculate_ios",
    "get_merger",
    "RTDetrV4ONNXModel",
    "RTDetrONNXModel",
    "RTDetrV2ONNXModel",
    "EngineConfig",
    "SlicingConfig",
    "InferenceConfig",
    "load_config",
    "export_batch_summary_csv",
    "IGNORED_CATEGORIES",
    "KNOWN_COLONY_LABELS",
    "compute_detection_metrics",
    "extract_gt_colony_boxes",
    "is_colony_label",
    "is_valid_highres_plate",
    "plot_plate_detections",
    "stream_sample_plate",
]
