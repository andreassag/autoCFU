"""End-to-end sliced inference, StarDist mask rasterization, and countability pipeline."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageDraw

from autocfu.config import EngineConfig
from autocfu.merger import Detection, get_merger
from autocfu.model import (
    AnyONNXModel,
    CountabilityONNXModel,
    CountabilityResult,
    load_onnx_model,
)
from autocfu.slicer import PlateSlicer

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


@dataclass
class PlateCountabilityReport:
    """Aggregated whole-plate and section-level countability assessment.

    Attributes:
        whole_plate: Classification result on the resized full dish image, if evaluated.
        section_predictions: Per-tile/section classification results, if evaluated.
        uncountable_threshold: Threshold used to determine binary countability.
    """

    whole_plate: CountabilityResult | None = None
    section_predictions: list[CountabilityResult] = field(default_factory=list)
    uncountable_threshold: float = 0.50

    @property
    def total_sections(self) -> int:
        """Number of evaluated plate sections."""
        return len(self.section_predictions)

    @property
    def empty_section_count(self) -> int:
        """Number of sections predicted as ``empty``."""
        return sum(1 for s in self.section_predictions if s.pred_label == "empty")

    @property
    def countable_section_count(self) -> int:
        """Number of sections predicted as ``countable``."""
        return sum(1 for s in self.section_predictions if s.pred_label == "countable")

    @property
    def uncountable_section_count(self) -> int:
        """Number of sections predicted as ``uncountable``."""
        return sum(1 for s in self.section_predictions if s.pred_label == "uncountable")

    @property
    def is_countable(self) -> bool:
        """Plate-level binary countability verdict.

        Uses ``whole_plate.is_countable`` when available; otherwise requires the mean
        section ``P(uncountable)`` to remain below ``uncountable_threshold`` (or zero
        evaluated sections).
        """
        if self.whole_plate is not None:
            return self.whole_plate.is_countable
        if not self.section_predictions:
            return True
        mean_unc = sum(s.uncountable_probability for s in self.section_predictions) / float(
            len(self.section_predictions)
        )
        return mean_unc < self.uncountable_threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-compatible dictionary."""
        return {
            "is_countable": self.is_countable,
            "uncountable_threshold": self.uncountable_threshold,
            "whole_plate": self.whole_plate.to_dict() if self.whole_plate is not None else None,
            "section_summary": {
                "total_sections": self.total_sections,
                "empty": self.empty_section_count,
                "countable": self.countable_section_count,
                "uncountable": self.uncountable_section_count,
            },
            "sections": [sec.to_dict() for sec in self.section_predictions],
        }


@dataclass
class InferenceResult:
    """Complete result returned by :meth:`AutoCFUEngine.predict`.

    Attributes:
        image_path: Source image path when available.
        image_size: Original dish dimensions ``(width, height)`` in pixels.
        model_type: Executed model family (``centernet``, ``stardist``, or ``countability``).
        detections: Deduplicated colony detections/polygons in global dish coordinates.
        num_slices: Number of slices processed.
        inference_time_ms: Total end-to-end execution latency in milliseconds.
        countability_report: Populated for ``countability`` models.
    """

    image_path: str | None
    image_size: tuple[int, int]
    model_type: str = "centernet"
    detections: list[Detection] = field(default_factory=list)
    num_slices: int = 1
    inference_time_ms: float = 0.0
    countability_report: PlateCountabilityReport | None = None

    @property
    def colony_count(self) -> int:
        """Total number of detected or segmented colonies."""
        return len(self.detections)

    @property
    def radii(self) -> list[float]:
        """Estimated radius (in pixels) for each detected colony."""
        return [det.radius_px if det.radius_px is not None else 0.0 for det in self.detections]

    @property
    def polygons(self) -> list[list[tuple[float, float]] | None]:
        """Polygon vertices for each detection (populated for StarDist models)."""
        return [det.polygon for det in self.detections]

    @property
    def is_countable(self) -> bool | None:
        """Binary countability verdict when running a Countability model, else ``None``."""
        if self.countability_report is None:
            return None
        return self.countability_report.is_countable

    def to_mask(self, dtype: np.dtype[Any] | type[Any] = np.uint16) -> np.ndarray:
        """Rasterize detected colonies/polygons into an instance segmentation mask.

        Background pixels have value ``0`` and each colony receives a unique integer
        instance ID in ``1..N`` (clamped to dtype max for ``uint16``). Uses star-convex
        polygon vertices when present (StarDist) or radial circles (CenterNet).

        Args:
            dtype: NumPy integer dtype for the output mask (defaults to ``np.uint16``).

        Returns:
            2D array of shape ``(height, width)``.
        """
        width, height = int(self.image_size[0]), int(self.image_size[1])
        canvas = Image.new("I", (width, height), 0)
        draw = ImageDraw.Draw(canvas)

        max_val = 65535 if np.dtype(dtype) == np.dtype(np.uint16) else 2_147_483_647
        for inst_id, det in enumerate(self.detections, start=1):
            fill_val = min(inst_id, max_val)
            if det.polygon and len(det.polygon) >= 3:
                draw.polygon(det.polygon, fill=fill_val)
            elif det.center_xy is not None and det.radius_px is not None and det.radius_px > 0.0:
                cx, cy = det.center_xy
                r = det.radius_px
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill_val)
            elif det.bbox is not None:
                draw.rectangle(det.bbox, fill=fill_val)

        return np.asarray(canvas, dtype=np.int32).astype(dtype)

    def save_mask(self, output_path: str | Path) -> Path:
        """Save the rasterized ``uint16`` instance segmentation mask to ``.tif`` or ``.png``.

        Args:
            output_path: Destination image path.

        Returns:
            Resolved output path.
        """
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        mask_u16 = self.to_mask(dtype=np.uint16)
        suffix = dest.suffix.lower()
        if suffix in (".tif", ".tiff"):
            Image.fromarray(mask_u16).save(dest, format="TIFF")
        else:
            Image.fromarray(mask_u16).save(dest, format="PNG")
        return dest

    def draw_overlay(
        self,
        image: np.ndarray | Image.Image | None = None,
        output_path: str | Path | None = None,
        *,
        show_labels: bool = False,
    ) -> Image.Image:
        """Render visual detections, StarDist contours, or countability grid onto the dish image.

        Args:
            image: Source dish image. Loaded from ``self.image_path`` if omitted.
            output_path: Optional destination file to write the rendered overlay.
            show_labels: Whether to annotate confidence scores next to detections.

        Returns:
            Annotated RGB ``PIL.Image.Image``.
        """
        base_img = self._resolve_base_image(image)
        if self.model_type == "countability" and self.countability_report is not None:
            annotated = self._draw_countability_overlay(base_img)
        else:
            annotated = self._draw_colony_overlay(base_img, show_labels=show_labels)

        if output_path is not None:
            dest = Path(output_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(dest)
        return annotated

    def _resolve_base_image(self, image: np.ndarray | Image.Image | None) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if image is not None:
            arr = np.asarray(image)
            if arr.dtype != np.uint8:
                scale = 255.0 if arr.size and float(np.max(arr)) <= 1.0 else 1.0
                arr = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        if self.image_path is not None and Path(self.image_path).exists():
            with Image.open(self.image_path) as handle:
                return handle.convert("RGB")
        return Image.new("RGB", (int(self.image_size[0]), int(self.image_size[1])), (0, 0, 0))

    def _draw_colony_overlay(self, base_img: Image.Image, *, show_labels: bool) -> Image.Image:
        canvas = base_img.copy()
        draw = ImageDraw.Draw(canvas)
        for det in self.detections:
            if det.polygon and len(det.polygon) >= 3:
                draw.polygon(det.polygon, outline=(0, 255, 160), width=2)
            elif det.center_xy is not None and det.radius_px is not None:
                cx, cy = det.center_xy
                r = max(1.0, det.radius_px)
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 230, 118), width=2)
            elif det.bbox is not None:
                draw.rectangle(det.bbox, outline=(0, 230, 118), width=2)

            if det.center_xy is not None:
                cx, cy = det.center_xy
                draw.ellipse((cx - 2.0, cy - 2.0, cx + 2.0, cy + 2.0), fill=(255, 82, 82))

            if show_labels and det.bbox is not None:
                draw.text((det.bbox[0], max(0.0, det.bbox[1] - 12.0)), f"{det.score:.2f}", fill=(255, 255, 0))
        return canvas

    def _draw_countability_overlay(self, base_img: Image.Image) -> Image.Image:
        report = self.countability_report
        if report is None:
            raise ValueError("Cannot render countability overlay when countability_report is None.")
        rgba = base_img.convert("RGBA")
        overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        palette = {
            "empty": ((52, 152, 219, 55), (52, 152, 219, 220)),
            "countable": ((46, 204, 113, 60), (46, 204, 113, 230)),
            "uncountable": ((231, 76, 60, 80), (231, 76, 60, 240)),
        }

        for sec in report.section_predictions:
            if sec.region_bbox is None:
                continue
            fill_col, edge_col = palette.get(sec.pred_label, ((200, 200, 200, 50), (255, 255, 255, 200)))
            x1, y1, x2, y2 = sec.region_bbox
            draw.rectangle((x1, y1, x2, y2), fill=fill_col, outline=edge_col, width=2)
            draw.text(
                (x1 + 6, y1 + 6),
                f"{sec.pred_label} ({sec.probs[sec.pred_class_id]:.2f})",
                fill=(255, 255, 255, 255),
            )

        verdict = "COUNTABLE" if report.is_countable else "UNCOUNTABLE"
        banner_bg = (39, 174, 96, 200) if report.is_countable else (192, 57, 43, 210)
        banner_text = f"{verdict}"
        if report.whole_plate is not None:
            wp = report.whole_plate
            banner_text += (
                f" | class={wp.pred_label} "
                f"(P_empty={wp.probs[0]:.2f}, P_countable={wp.probs[1]:.2f}, P_unc={wp.probs[2]:.2f})"
            )
        draw.rectangle((0, 0, rgba.width, 32), fill=banner_bg)
        draw.text((10, 8), banner_text, fill=(255, 255, 255, 255))
        return Image.alpha_composite(rgba, overlay).convert("RGB")

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete inference results to a JSON-compatible dictionary."""
        payload: dict[str, Any] = {
            "image_path": self.image_path,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "model_type": self.model_type,
            "colony_count": self.colony_count,
            "num_slices": self.num_slices,
            "inference_time_ms": round(self.inference_time_ms, 3),
        }
        if self.countability_report is not None:
            payload["countability"] = self.countability_report.to_dict()
        if self.model_type != "countability" or self.detections:
            payload["detections"] = [det.to_dict() for det in self.detections]
        return payload

    def save_json(self, output_path: str | Path) -> Path:
        """Write inference results to a formatted JSON file."""
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return dest

    def save_csv(self, output_path: str | Path) -> Path:
        """Write detections or countability predictions to a CSV file."""
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with dest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if self.model_type == "countability" and self.countability_report is not None:
                writer.writerow(
                    [
                        "region_type",
                        "xmin",
                        "ymin",
                        "xmax",
                        "ymax",
                        "pred_class_id",
                        "pred_label",
                        "is_countable",
                        "prob_empty",
                        "prob_countable",
                        "prob_uncountable",
                    ]
                )
                if self.countability_report.whole_plate is not None:
                    wp = self.countability_report.whole_plate
                    writer.writerow(
                        [
                            "whole_plate",
                            0,
                            0,
                            self.image_size[0],
                            self.image_size[1],
                            wp.pred_class_id,
                            wp.pred_label,
                            wp.is_countable,
                            f"{wp.probs[0]:.6f}",
                            f"{wp.probs[1]:.6f}",
                            f"{wp.probs[2]:.6f}",
                        ]
                    )
                for sec in self.countability_report.section_predictions:
                    bx = sec.region_bbox or (0, 0, self.image_size[0], self.image_size[1])
                    writer.writerow(
                        [
                            "section",
                            bx[0],
                            bx[1],
                            bx[2],
                            bx[3],
                            sec.pred_class_id,
                            sec.pred_label,
                            sec.is_countable,
                            f"{sec.probs[0]:.6f}",
                            f"{sec.probs[1]:.6f}",
                            f"{sec.probs[2]:.6f}",
                        ]
                    )
            else:
                writer.writerow(
                    [
                        "instance_id",
                        "label",
                        "class_id",
                        "score",
                        "cx",
                        "cy",
                        "radius_px",
                        "xmin",
                        "ymin",
                        "xmax",
                        "ymax",
                    ]
                )
                for idx, det in enumerate(self.detections, start=1):
                    if det.bbox is None or det.center_xy is None:
                        raise ValueError(f"Detection {idx} is missing required bbox or center_xy coordinates.")
                    writer.writerow(
                        [
                            idx,
                            det.label,
                            det.class_id,
                            f"{det.score:.6f}",
                            f"{det.center_xy[0]:.3f}",
                            f"{det.center_xy[1]:.3f}",
                            f"{(det.radius_px or 0.0):.3f}",
                            f"{det.bbox[0]:.3f}",
                            f"{det.bbox[1]:.3f}",
                            f"{det.bbox[2]:.3f}",
                            f"{det.bbox[3]:.3f}",
                        ]
                    )
        return dest


def _open_rgb_image(image: str | Path | np.ndarray | Image.Image) -> tuple[Image.Image, str | None]:
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")
        with Image.open(path) as handle:
            return handle.convert("RGB"), str(path)
    if isinstance(image, Image.Image):
        return image.convert("RGB"), None
    arr = np.asarray(image)
    if arr.ndim == 2:
        pil_img = Image.fromarray(arr).convert("RGB")
    else:
        if np.issubdtype(arr.dtype, np.floating):
            scale = 255.0 if arr.size and float(np.max(arr)) <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)
        pil_img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return pil_img, None


class AutoCFUEngine:
    """High-resolution sliced inference orchestrator for AutoCFU ONNX models.

    Automatically dispatches execution based on the loaded ONNX model family:

    - ``centernet`` / ``stardist``: Slices large plates via :class:`PlateSlicer`, runs
      per-tile ONNX prediction, maps local coordinates to dish coordinates, and
      merges seam duplicates via ``GreedyNMM`` or ``NMS``.
    - ``countability``: Evaluates whole-plate countability and/or 640x640 section
      countability according to ``config.inference.countability_scope``.
    """

    def __init__(
        self,
        model_path: str | Path,
        config: EngineConfig | None = None,
        *,
        model: AnyONNXModel | None = None,
    ) -> None:
        self.config = config if config is not None else EngineConfig()
        inf = self.config.inference
        sli = self.config.slicing

        self.model: AnyONNXModel = (
            model
            if model is not None
            else load_onnx_model(
                model_path=model_path,
                model_type=inf.model_type,
                device=inf.device,
                input_size=inf.input_size,
                top_k=inf.top_k,
            )
        )
        self.model_type = self.model.model_type
        self.slicer = PlateSlicer(
            slice_width=sli.slice_width,
            slice_height=sli.slice_height,
            overlap_width_ratio=sli.overlap_width_ratio,
            overlap_height_ratio=sli.overlap_height_ratio,
        )
        self.merger = get_merger(
            merger_type=inf.merger_type,
            match_threshold=inf.postprocess_match_threshold,
            match_metric=inf.match_metric,
        )

    def predict(
        self,
        image: str | Path | np.ndarray | Image.Image,
        *,
        image_path: str | None = None,
    ) -> InferenceResult:
        """Run full-plate inference on a single Petri dish image or array."""
        t0 = time.perf_counter()
        pil_img, resolved_path = _open_rgb_image(image)
        if image_path is not None:
            resolved_path = image_path

        width, height = pil_img.size
        if self.model_type == "countability" or isinstance(self.model, CountabilityONNXModel):
            result = self._predict_countability(pil_img, resolved_path, width, height)
        else:
            result = self._predict_spatial(pil_img, resolved_path, width, height)

        result.inference_time_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def _predict_countability(
        self,
        pil_img: Image.Image,
        image_path: str | None,
        width: int,
        height: int,
    ) -> InferenceResult:
        model = cast(CountabilityONNXModel, self.model)
        inf = self.config.inference
        scope = inf.countability_scope
        whole_plate_res: CountabilityResult | None = None
        section_results: list[CountabilityResult] = []
        num_slices = 0

        if scope in ("both", "plate"):
            whole_plate_res = model.predict(
                pil_img,
                uncountable_threshold=inf.uncountable_threshold,
                region_bbox=(0, 0, width, height),
            )
            num_slices += 1

        if scope in ("both", "sections"):
            windows = self.slicer.get_slice_windows(width, height)
            for win in windows:
                crop = pil_img.crop((win.xmin, win.ymin, win.xmax, win.ymax))
                sec_res = model.predict(
                    crop,
                    uncountable_threshold=inf.uncountable_threshold,
                    region_bbox=(win.xmin, win.ymin, win.xmax, win.ymax),
                )
                section_results.append(sec_res)
            num_slices += len(windows)

        report = PlateCountabilityReport(
            whole_plate=whole_plate_res,
            section_predictions=section_results,
            uncountable_threshold=inf.uncountable_threshold,
        )
        return InferenceResult(
            image_path=image_path,
            image_size=(width, height),
            model_type="countability",
            detections=[],
            num_slices=max(1, num_slices),
            countability_report=report,
        )

    def _predict_tile(self, tile: Image.Image) -> list[Detection]:
        """Run the loaded spatial colony model on a single crop or tile."""
        model = cast("Any", self.model)
        inf = self.config.inference
        return model.predict(
            tile,
            conf_threshold=inf.confidence_threshold,
            min_radius=inf.min_radius,
            max_radius=inf.max_radius,
            min_box_size=inf.min_box_size,
            max_box_size=inf.max_box_size,
            max_aspect_ratio=inf.max_aspect_ratio,
            target_class_id=inf.target_class_id,
            class_name=inf.class_name,
            intra_tile_nms_threshold=inf.intra_tile_nms_threshold,
            intra_tile_metric="ios",
        )

    def _predict_spatial(
        self,
        pil_img: Image.Image,
        image_path: str | None,
        width: int,
        height: int,
    ) -> InferenceResult:
        sli = self.config.slicing

        # Avoid slicing overhead if image already fits within one slice window
        if width <= sli.slice_width and height <= sli.slice_height:
            detections = self._predict_tile(pil_img)
            if self.merger is not None:
                detections = self.merger.merge(detections)
            return InferenceResult(
                image_path=image_path,
                image_size=(width, height),
                model_type=self.model_type,
                detections=detections,
                num_slices=1,
            )

        windows = self.slicer.get_slice_windows(width, height)
        global_detections: list[Detection] = []
        for win in windows:
            crop = pil_img.crop((win.xmin, win.ymin, win.xmax, win.ymax))
            tile_dets = self._predict_tile(crop)
            for det in tile_dets:
                global_detections.append(det.translated(float(win.xmin), float(win.ymin)))

        if self.merger is not None:
            merged = self.merger.merge(global_detections)
        else:
            merged = global_detections

        return InferenceResult(
            image_path=image_path,
            image_size=(width, height),
            model_type=self.model_type,
            detections=merged,
            num_slices=len(windows),
        )

    def predict_batch(
        self,
        image_paths: Sequence[str | Path],
        *,
        output_dir: str | Path | None = None,
        save_overlay: bool = False,
        save_json: bool = False,
        save_csv: bool = False,
        save_mask: bool = False,
        show_labels: bool = False,
    ) -> list[InferenceResult]:
        """Run inference across multiple images and optionally write outputs per image."""
        results: list[InferenceResult] = []
        out_root = Path(output_dir) if output_dir is not None else None
        if out_root is not None and (save_overlay or save_json or save_csv or save_mask):
            out_root.mkdir(parents=True, exist_ok=True)

        for item in image_paths:
            src_path = Path(item)
            res = self.predict(src_path)
            results.append(res)

            if out_root is not None:
                stem = src_path.stem
                if save_overlay:
                    res.draw_overlay(output_path=out_root / f"{stem}_overlay.png", show_labels=show_labels)
                if save_json:
                    res.save_json(out_root / f"{stem}_predictions.json")
                if save_csv:
                    res.save_csv(out_root / f"{stem}_predictions.csv")
                if save_mask and res.model_type != "countability":
                    res.save_mask(out_root / f"{stem}_mask.tif")

        return results


def export_batch_summary_csv(results: Sequence[InferenceResult], output_path: str | Path) -> Path:
    """Export an image-level summary CSV across multiple inference results."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_path",
                "model_type",
                "width",
                "height",
                "colony_count",
                "mean_radius_px",
                "is_countable",
                "pred_label",
                "num_slices",
                "inference_time_ms",
            ]
        )
        for res in results:
            radii = res.radii
            mean_radius = sum(radii) / float(len(radii)) if radii else 0.0
            pred_label = (
                res.countability_report.whole_plate.pred_label
                if res.countability_report is not None and res.countability_report.whole_plate is not None
                else ""
            )
            writer.writerow(
                [
                    res.image_path or "",
                    res.model_type,
                    res.image_size[0],
                    res.image_size[1],
                    res.colony_count,
                    f"{mean_radius:.3f}",
                    "" if res.is_countable is None else res.is_countable,
                    pred_label,
                    res.num_slices,
                    f"{res.inference_time_ms:.2f}",
                ]
            )
    return dest
