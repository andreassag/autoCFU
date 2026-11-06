"""End-to-end full plate sliced inference pipeline."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import EngineConfig, load_config
from .merger import Detection, get_merger
from .model import RTDetrV2ONNXModel
from .slicer import PlateSlicer


@dataclass
class InferenceResult:
    """Full-dish inference results, detections, and execution statistics."""

    image_path: str | None
    image_size: tuple[int, int]  # (width, height)
    detections: list[Detection]
    colony_count: int
    num_slices: int
    execution_time_seconds: float
    raw_detections: list[Detection] | None = None

    @property
    def bboxes(self) -> list[tuple[float, float, float, float]]:
        """Return list of bounding boxes [(x1, y1, x2, y2), ...] in global coordinates."""
        return [d.bbox for d in self.detections]

    @property
    def centers(self) -> list[tuple[float, float]]:
        """Return list of colony centroid coordinates [(cx, cy), ...] in global coordinates."""
        return [d.center for d in self.detections]

    @property
    def scores(self) -> list[float]:
        """Return list of confidence scores."""
        return [d.score for d in self.detections]

    def to_numpy(self, include_score: bool = True) -> np.ndarray:
        """Export detections as a NumPy array of shape [N, 5] (x1, y1, x2, y2, score) or [N, 4]."""
        if not self.detections:
            return np.empty((0, 5 if include_score else 4), dtype=np.float32)
        if include_score:
            return np.array([[d.x1, d.y1, d.x2, d.y2, d.score] for d in self.detections], dtype=np.float32)
        return np.array([d.bbox for d in self.detections], dtype=np.float32)

    def centers_numpy(self) -> np.ndarray:
        """Export colony centroids as a NumPy array of shape [N, 2] (cx, cy)."""
        if not self.detections:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([d.center for d in self.detections], dtype=np.float32)

    def filter_by_roi(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        by_center: bool = True,
    ) -> list[Detection]:
        """Return detections located within a rectangular region of interest [x1, y1, x2, y2]."""
        matches = []
        for det in self.detections:
            if by_center:
                cx, cy = det.center
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    matches.append(det)
            else:
                if det.x1 >= x1 and det.y1 >= y1 and det.x2 <= x2 and det.y2 <= y2:
                    matches.append(det)
        return matches

    def filter_by_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
    ) -> list[Detection]:
        """Return detections whose center falls within a circle (e.g. a pipetted drop spot)."""
        r_sq = radius * radius
        matches = []
        for det in self.detections:
            dcx, dcy = det.center
            if (dcx - cx) ** 2 + (dcy - cy) ** 2 <= r_sq:
                matches.append(det)
        return matches

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "image_path": self.image_path,
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
            "colony_count": self.colony_count,
            "num_slices": self.num_slices,
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "detections": [det.to_dict() for det in self.detections],
        }
        if self.raw_detections is not None:
            data["raw_detections"] = [det.to_dict() for det in self.raw_detections]
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, output_path: str | Path, indent: int = 2) -> None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))

    def save_csv(self, output_path: str | Path) -> None:
        """Write detections bounding boxes and scores to CSV."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["detection_index", "class_id", "label", "score", "x1", "y1", "x2", "y2"])
            for idx, det in enumerate(self.detections):
                writer.writerow(
                    [
                        idx + 1,
                        det.class_id,
                        det.label,
                        f"{det.score:.4f}",
                        f"{det.x1:.2f}",
                        f"{det.y1:.2f}",
                        f"{det.x2:.2f}",
                        f"{det.y2:.2f}",
                    ]
                )

    def draw_overlay(
        self,
        output_path: str | Path | None = None,
        image: Image.Image | np.ndarray | None = None,
        box_color: tuple[int, int, int] = (255, 0, 0),
        box_width: int = 3,
        show_score: bool = True,
    ) -> Image.Image:
        """Draw bounding box overlays on the original plate image."""
        if image is not None:
            if isinstance(image, np.ndarray):
                pil_img = Image.fromarray(image).convert("RGB")
            else:
                pil_img = image.convert("RGB")
        elif self.image_path is not None and Path(self.image_path).is_file():
            pil_img = Image.open(self.image_path).convert("RGB")
        else:
            # Create a blank canvas matching image_size if no source image provided
            pil_img = Image.new("RGB", self.image_size, color=(240, 240, 240))

        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.load_default()

        for det in self.detections:
            draw.rectangle(
                [det.x1, det.y1, det.x2, det.y2],
                outline=box_color,
                width=box_width,
            )
            if show_score:
                label_text = f"{det.score:.2f}"
                draw.text(
                    (det.x1 + 2, max(0, det.y1 - 10)),
                    label_text,
                    fill=box_color,
                    font=font,
                )

        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(out_file)

        return pil_img


class AutoCFUEngine:
    """Orchestrates high-resolution sliced colony detection across entire Petri dishes."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        model: Any | None = None,
        config: EngineConfig | str | Path | None = None,
        device: str = "auto",
    ) -> None:
        if isinstance(config, (str, Path)):
            self.config = load_config(config)
        elif isinstance(config, EngineConfig):
            self.config = config
        else:
            self.config = load_config(None)

        self.slicer = PlateSlicer.from_config(self.config.slicing)
        self.merger = get_merger(
            merger_type=self.config.inference.merger_type,
            iou_threshold=self.config.inference.postprocess_match_threshold,
            match_metric=self.config.inference.match_metric,
        )

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = RTDetrV2ONNXModel(
                model_path=model_path,
                device=device,
                input_size=self.config.slicing.slice_width,
            )
        else:
            raise ValueError("Either 'model' or 'model_path' must be provided.")

    def infer_image(
        self,
        image_input: str | Path | Image.Image | np.ndarray,
        return_raw: bool = False,
    ) -> InferenceResult:
        """Run sliced inference on a single full-resolution plate."""
        start_time = time.perf_counter()
        image_path: str | None = None

        if isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            pil_img = Image.open(image_path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        orig_w, orig_h = pil_img.size

        # Generate tiles and their absolute slice window offsets
        slices = self.slicer.slice_image(pil_img)
        raw_detections: list[Detection] = []

        for window, crop in slices:
            slice_dets = self.model.predict_slice(
                crop=crop,
                confidence_threshold=self.config.inference.confidence_threshold,
                target_class_id=self.config.inference.target_class_id,
                class_name=self.config.inference.class_name,
            )

            # Translate slice coordinates into full-plate coordinate space
            for det in slice_dets:
                translated = Detection(
                    bbox=(
                        det.x1 + window.xmin,
                        det.y1 + window.ymin,
                        det.x2 + window.xmin,
                        det.y2 + window.ymin,
                    ),
                    score=det.score,
                    class_id=det.class_id,
                    label=det.label,
                )
                raw_detections.append(translated)

        # Merge overlapping detections from adjacent tiles
        merged_detections = self.merger.merge(raw_detections)
        elapsed = time.perf_counter() - start_time

        return InferenceResult(
            image_path=image_path,
            image_size=(orig_w, orig_h),
            detections=merged_detections,
            colony_count=len(merged_detections),
            num_slices=len(slices),
            execution_time_seconds=elapsed,
            raw_detections=raw_detections if return_raw else None,
        )

    def infer_batch(
        self,
        image_paths: Sequence[str | Path],
    ) -> list[InferenceResult]:
        """Run inference across a list of image paths."""
        results: list[InferenceResult] = []
        for path in image_paths:
            results.append(self.infer_image(path))
        return results


def export_batch_summary_csv(results: list[InferenceResult], output_csv_path: str | Path) -> None:
    """Save batch inference summary counts to a single CSV file."""
    target = Path(output_csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "width", "height", "colony_count", "num_slices", "time_sec"])
        for res in results:
            writer.writerow(
                [
                    res.image_path or "unknown",
                    res.image_size[0],
                    res.image_size[1],
                    res.colony_count,
                    res.num_slices,
                    f"{res.execution_time_seconds:.4f}",
                ]
            )


# Backward-compatibility alias
AgarInferenceEngine = AutoCFUEngine
