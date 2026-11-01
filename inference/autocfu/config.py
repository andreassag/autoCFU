"""Configuration dataclasses and YAML loading utilities for AutoCFU."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


@dataclass
class SlicingConfig:
    """Parameters controlling 2D grid image slicing."""

    slice_width: int = 640
    slice_height: int = 640
    overlap_width_ratio: float = 0.20
    overlap_height_ratio: float = 0.20

    @property
    def step_x(self) -> int:
        """Horizontal step size in pixels."""
        step = int(self.slice_width * (1.0 - self.overlap_width_ratio))
        return max(1, step)

    @property
    def step_y(self) -> int:
        """Vertical step size in pixels."""
        step = int(self.slice_height * (1.0 - self.overlap_height_ratio))
        return max(1, step)


@dataclass
class InferenceConfig:
    """Parameters controlling model execution and post-processing merging."""

    confidence_threshold: float = 0.35
    postprocess_match_threshold: float = 0.50
    merger_type: str = "greedy_nmm"  # Options: 'greedy_nmm', 'nms'
    match_metric: str = "iou"  # Options: 'iou', 'ios'
    target_class_id: int = 0
    class_name: str = "colony"


@dataclass
class EngineConfig:
    """Top-level configuration for the inference engine."""

    slicing: SlicingConfig = field(default_factory=SlicingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "slicing": {
                "slice_width": self.slicing.slice_width,
                "slice_height": self.slicing.slice_height,
                "overlap_width_ratio": self.slicing.overlap_width_ratio,
                "overlap_height_ratio": self.slicing.overlap_height_ratio,
            },
            "inference": {
                "confidence_threshold": self.inference.confidence_threshold,
                "postprocess_match_threshold": self.inference.postprocess_match_threshold,
                "merger_type": self.inference.merger_type,
                "match_metric": self.inference.match_metric,
                "target_class_id": self.inference.target_class_id,
                "class_name": self.inference.class_name,
            },
        }

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Serialize configuration to a YAML file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_source: Optional[Union[str, Path, Dict[str, Any]]] = None) -> EngineConfig:
    """Load EngineConfig from a YAML file path or dictionary.

    If config_source is None, default settings are returned.
    """
    if config_source is None:
        return EngineConfig()

    data: Dict[str, Any]
    if isinstance(config_source, (str, Path)):
        path = Path(config_source)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif isinstance(config_source, dict):
        data = config_source
    else:
        raise TypeError(f"Unsupported config source type: {type(config_source)}")

    slicing_data = data.get("slicing", {})
    inference_data = data.get("inference", {})

    slicing_cfg = SlicingConfig(
        slice_width=int(slicing_data.get("slice_width", 640)),
        slice_height=int(slicing_data.get("slice_height", 640)),
        overlap_width_ratio=float(slicing_data.get("overlap_width_ratio", 0.20)),
        overlap_height_ratio=float(slicing_data.get("overlap_height_ratio", 0.20)),
    )

    inference_cfg = InferenceConfig(
        confidence_threshold=float(inference_data.get("confidence_threshold", 0.35)),
        postprocess_match_threshold=float(inference_data.get("postprocess_match_threshold", 0.50)),
        merger_type=str(inference_data.get("merger_type", "greedy_nmm")),
        match_metric=str(inference_data.get("match_metric", "iou")),
        target_class_id=int(inference_data.get("target_class_id", 0)),
        class_name=str(inference_data.get("class_name", "colony")),
    )

    return EngineConfig(slicing=slicing_cfg, inference=inference_cfg)
