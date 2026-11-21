"""Configuration schema and YAML loading for the AutoCFU inference engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_MODEL_TYPES: tuple[str, ...] = ("auto", "centernet", "stardist", "countability")
VALID_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")
VALID_MERGER_TYPES: tuple[str, ...] = ("greedy_nmm", "nms", "none")
VALID_MATCH_METRICS: tuple[str, ...] = ("center_dist", "ios", "iou")
VALID_COUNTABILITY_SCOPES: tuple[str, ...] = ("both", "plate", "sections")


@dataclass
class SlicingConfig:
    """Configuration for 2D sliding-window dish slicing.

    Attributes:
        slice_width: Width of each square/rectangular image slice in pixels.
        slice_height: Height of each image slice in pixels.
        overlap_width_ratio: Horizontal fractional overlap between adjacent tiles in ``[0.0, 1.0)``.
        overlap_height_ratio: Vertical fractional overlap between adjacent tiles in ``[0.0, 1.0)``.
    """

    slice_width: int = 640
    slice_height: int = 640
    overlap_width_ratio: float = 0.20
    overlap_height_ratio: float = 0.20

    def __post_init__(self) -> None:
        if self.slice_width <= 0 or self.slice_height <= 0:
            raise ValueError(f"Slice dimensions must be positive, got {self.slice_width}x{self.slice_height}.")
        if not (0.0 <= self.overlap_width_ratio < 1.0):
            raise ValueError(f"overlap_width_ratio must be in [0.0, 1.0), got {self.overlap_width_ratio}.")
        if not (0.0 <= self.overlap_height_ratio < 1.0):
            raise ValueError(f"overlap_height_ratio must be in [0.0, 1.0), got {self.overlap_height_ratio}.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize slicing configuration to a plain dictionary."""
        return {
            "slice_width": self.slice_width,
            "slice_height": self.slice_height,
            "overlap_width_ratio": self.overlap_width_ratio,
            "overlap_height_ratio": self.overlap_height_ratio,
        }


@dataclass
class InferenceConfig:
    """Configuration for ONNX inference, geometric filtering, merging, and countability.

    Attributes:
        model_type: Model architecture family (``auto``, ``centernet``, ``stardist``, or ``countability``).
        input_size: Square model input resolution in pixels.
        device: Execution provider preference (``auto``, ``cpu``, or ``cuda``).
        confidence_threshold: Minimum confidence score for retaining predictions.
        postprocess_match_threshold: Similarity threshold used by cross-tile merging.
        merger_type: Merging strategy (``greedy_nmm``, ``nms``, or ``none``).
        match_metric: Similarity metric for merging (``center_dist``, ``ios``, or ``iou``).
        intra_tile_nms_threshold: Match threshold applied within each tile prior to global merging.
        min_radius: Minimum valid radial colony size in pixels.
        max_radius: Maximum valid radial colony size in pixels.
        min_box_size: Minimum valid bounding-box width or height in pixels.
        max_box_size: Maximum valid bounding-box width or height in pixels.
        max_aspect_ratio: Maximum allowed bounding-box aspect ratio (``max(w/h, h/w)``).
        target_class_id: Target foreground class index for single-class colony models.
        class_name: Human-readable foreground label assigned to detections.
        top_k: Maximum number of raw peak candidates decoded per tile.
        num_rays: Number of equiangular rays expected for 2D StarDist models.
        countability_scope: Evaluation mode for countability models (``both``, ``plate``, or ``sections``).
        uncountable_threshold: Threshold on ``P(uncountable)`` above which a plate/section is marked uncountable.
    """

    model_type: str = "auto"
    input_size: int = 640
    device: str = "auto"
    confidence_threshold: float = 0.35
    postprocess_match_threshold: float = 0.50
    merger_type: str = "greedy_nmm"
    match_metric: str = "center_dist"
    intra_tile_nms_threshold: float = 0.45
    min_radius: float = 1.5
    max_radius: float = 250.0
    min_box_size: float = 3.0
    max_box_size: float = 500.0
    max_aspect_ratio: float = 4.0
    target_class_id: int = 0
    class_name: str = "colony"
    top_k: int = 300
    num_rays: int = 32
    countability_scope: str = "both"
    uncountable_threshold: float = 0.50

    def __post_init__(self) -> None:
        self.model_type = self.model_type.strip().lower()
        self.merger_type = self.merger_type.strip().lower()
        self.match_metric = self.match_metric.strip().lower()
        self.countability_scope = self.countability_scope.strip().lower()
        self.device = self.device.strip().lower()

        if self.model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"Unsupported model_type '{self.model_type}'. Expected one of {VALID_MODEL_TYPES}.")
        if self.device not in VALID_DEVICES:
            raise ValueError(f"Unsupported device '{self.device}'. Expected one of {VALID_DEVICES}.")
        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}.")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0.0, 1.0], got {self.confidence_threshold}.")
        if not (0.0 <= self.postprocess_match_threshold <= 2.0):
            raise ValueError(
                f"postprocess_match_threshold must be in [0.0, 2.0], got {self.postprocess_match_threshold}."
            )
        if self.merger_type not in VALID_MERGER_TYPES:
            raise ValueError(f"Unsupported merger_type '{self.merger_type}'. Expected one of {VALID_MERGER_TYPES}.")
        if self.match_metric not in VALID_MATCH_METRICS:
            raise ValueError(f"Unsupported match_metric '{self.match_metric}'. Expected one of {VALID_MATCH_METRICS}.")
        if not (0.0 <= self.intra_tile_nms_threshold <= 2.0):
            raise ValueError(f"intra_tile_nms_threshold must be in [0.0, 2.0], got {self.intra_tile_nms_threshold}.")
        if self.min_radius < 0.0 or self.max_radius <= self.min_radius:
            raise ValueError(f"Invalid radius bounds: min_radius={self.min_radius}, max_radius={self.max_radius}.")
        if self.min_box_size < 0.0 or self.max_box_size <= self.min_box_size:
            raise ValueError(
                f"Invalid box size bounds: min_box_size={self.min_box_size}, max_box_size={self.max_box_size}."
            )
        if self.max_aspect_ratio < 1.0:
            raise ValueError(f"max_aspect_ratio must be >= 1.0, got {self.max_aspect_ratio}.")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}.")
        if self.num_rays <= 0:
            raise ValueError(f"num_rays must be positive, got {self.num_rays}.")
        if self.countability_scope not in VALID_COUNTABILITY_SCOPES:
            raise ValueError(
                f"Unsupported countability_scope '{self.countability_scope}'. "
                f"Expected one of {VALID_COUNTABILITY_SCOPES}."
            )
        if not (0.0 <= self.uncountable_threshold <= 1.0):
            raise ValueError(f"uncountable_threshold must be in [0.0, 1.0], got {self.uncountable_threshold}.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize inference configuration to a plain dictionary."""
        return {
            "model_type": self.model_type,
            "input_size": self.input_size,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "postprocess_match_threshold": self.postprocess_match_threshold,
            "merger_type": self.merger_type,
            "match_metric": self.match_metric,
            "intra_tile_nms_threshold": self.intra_tile_nms_threshold,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "min_box_size": self.min_box_size,
            "max_box_size": self.max_box_size,
            "max_aspect_ratio": self.max_aspect_ratio,
            "target_class_id": self.target_class_id,
            "class_name": self.class_name,
            "top_k": self.top_k,
            "num_rays": self.num_rays,
            "countability_scope": self.countability_scope,
            "uncountable_threshold": self.uncountable_threshold,
        }


@dataclass
class EngineConfig:
    """Top-level configuration combining slicing and inference settings.

    Attributes:
        slicing: Tile generation parameters.
        inference: Model selection, filtering, merging, and countability settings.
    """

    slicing: SlicingConfig = field(default_factory=SlicingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EngineConfig:
        """Create an :class:`EngineConfig` from a nested dictionary.

        Args:
            raw: Mapping containing optional ``slicing`` and ``inference`` sections.

        Returns:
            Validated :class:`EngineConfig` instance.
        """
        slicing_data = dict(raw.get("slicing") or {})
        inference_data = dict(raw.get("inference") or {})
        return cls(
            slicing=SlicingConfig(**slicing_data),
            inference=InferenceConfig(**inference_data),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete engine configuration to a dictionary."""
        return {
            "slicing": self.slicing.to_dict(),
            "inference": self.inference.to_dict(),
        }


def load_config(path: str | Path | None = None) -> EngineConfig:
    """Load an :class:`EngineConfig` from a YAML file or fall back to defaults.

    Args:
        path: Optional path to a YAML configuration file. If ``None``, returns defaults.

    Returns:
        Validated :class:`EngineConfig` instance.

    Raises:
        FileNotFoundError: If ``path`` is provided but does not exist.
        ValueError: If the YAML root is not a mapping or contains invalid settings.
    """
    if path is None:
        return EngineConfig()

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return EngineConfig()
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at root of {cfg_path}, got {type(data).__name__}.")

    return EngineConfig.from_dict(data)
