"""AutoCFU: High-resolution sliced microbial colony detection engine."""

from __future__ import annotations

from .config import EngineConfig, InferenceConfig, SlicingConfig, load_config
from .merger import NMS, Detection, GreedyNMM, calculate_ios, calculate_iou, get_merger
from .model import RTDetrV2ONNXModel
from .pipeline import AgarInferenceEngine, AutoCFUEngine, InferenceResult, export_batch_summary_csv
from .slicer import PlateSlicer, SliceWindow

__version__ = "0.1.0"

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
    "RTDetrV2ONNXModel",
    "EngineConfig",
    "SlicingConfig",
    "InferenceConfig",
    "load_config",
    "export_batch_summary_csv",
]
