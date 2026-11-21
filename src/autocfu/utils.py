"""Utility helpers for dish label validation, ground-truth extraction, metrics, and plotting."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from autocfu.merger import Detection, calculate_iou

KNOWN_COLONY_LABELS: set[str] = {
    "colony",
    "colonies",
    "b.subtilis",
    "c.albicans",
    "e.coli",
    "p.aeruginosa",
    "s.aureus",
    "microbial_colony",
}

IGNORED_CATEGORIES: set[str] = {
    "",
    "background",
    "contamination",
    "defect",
    "noise",
    "bubble",
    "edge",
    "empty",
}


def is_colony_label(label: str | int | None) -> bool:
    """Return ``True`` when ``label`` represents a valid microbial colony target.

    Accepts non-negative integer class IDs and non-empty strings not present in
    ``IGNORED_CATEGORIES``.
    """
    if label is None:
        return False
    if isinstance(label, int):
        return label >= 0
    normalized = str(label).strip().lower()
    return bool(normalized) and normalized not in IGNORED_CATEGORIES


def is_valid_highres_plate(sample: dict[str, Any], min_size: int = 1000) -> bool:
    """Check whether a sample mapping represents a full-resolution dish image.

    Args:
        sample: Dictionary containing image metadata (``width``/``height``) or ``image``.
        min_size: Minimum width and height in pixels.

    Returns:
        ``True`` if dimensions satisfy ``width >= min_size`` and ``height >= min_size``
        and the sample is not flagged as a cropped tile.
    """
    if sample.get("is_tiled") is True:
        return False

    width = sample.get("width")
    height = sample.get("height")
    if (width is None or height is None) and "image" in sample:
        img = sample["image"]
        if isinstance(img, Image.Image):
            width, height = img.size
        elif isinstance(img, np.ndarray) and img.ndim >= 2:
            height, width = int(img.shape[0]), int(img.shape[1])

    if width is None or height is None:
        return False
    return int(width) >= int(min_size) and int(height) >= int(min_size)


def extract_gt_colony_boxes(sample: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    """Extract ground-truth ``(xmin, ymin, xmax, ymax)`` colony boxes from a sample record.

    Supports both ``centernet.json`` format (``objects`` containing ``x_px``, ``y_px``,
    ``r_px``) and generic annotation lists (``annotations`` or ``boxes``).

    Args:
        sample: Metadata dictionary or JSON payload.

    Returns:
        List of bounding boxes in ``(xmin, ymin, xmax, ymax)`` pixel coordinates.
    """
    boxes: list[tuple[float, float, float, float]] = []

    if isinstance(sample.get("objects"), list):
        for obj in sample["objects"]:
            if not isinstance(obj, dict):
                continue
            if obj.get("is_contamination") is True:
                continue
            label = obj.get("single_class_name", obj.get("class_name", "colony"))
            if not is_colony_label(label):
                continue
            if "x_px" in obj and "y_px" in obj and "r_px" in obj:
                cx = float(obj["x_px"])
                cy = float(obj["y_px"])
                r = max(0.0, float(obj["r_px"]))
                boxes.append((cx - r, cy - r, cx + r, cy + r))
        return boxes

    raw_items = sample.get("annotations") or sample.get("boxes") or []
    for item in raw_items:
        if isinstance(item, dict):
            label = item.get("category", item.get("label", "colony"))
            if not is_colony_label(label):
                continue
            coords = item.get("bbox") or item.get("box")
            if coords and len(coords) == 4:
                x1, y1, v3, v4 = (float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))
                fmt = str(item.get("format", "xyxy")).lower()
                if fmt == "xywh":
                    boxes.append((x1, y1, x1 + v3, y1 + v4))
                else:
                    boxes.append((x1, y1, v3, v4))
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            boxes.append((float(item[0]), float(item[1]), float(item[2]), float(item[3])))

    return boxes


def compute_detection_metrics(
    preds: Sequence[Detection],
    gt_boxes: Sequence[tuple[float, float, float, float]],
    iou_threshold: float = 0.50,
) -> dict[str, float]:
    """Compute greedy 1-to-1 IoU detection metrics against ground-truth boxes.

    Args:
        preds: Predicted :class:`Detection` instances.
        gt_boxes: Ground-truth ``(xmin, ymin, xmax, ymax)`` boxes.
        iou_threshold: Minimum IoU required for a true-positive match.

    Returns:
        Dictionary with ``precision``, ``recall``, ``f1``, ``tp``, ``fp``, ``fn``,
        ``pred_count``, ``gt_count``, and ``count_error``.
    """
    sorted_preds = sorted(preds, key=lambda d: -d.score)
    matched_gt = [False] * len(gt_boxes)
    tp = 0

    for det in sorted_preds:
        if det.bbox is None:
            continue
        best_iou = 0.0
        best_idx = -1
        for idx, gt_box in enumerate(gt_boxes):
            if matched_gt[idx]:
                continue
            iou = calculate_iou(det.bbox, gt_box)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            matched_gt[best_idx] = True
            tp += 1

    fp = len(sorted_preds) - tp
    fn = len(gt_boxes) - tp
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "pred_count": float(len(sorted_preds)),
        "gt_count": float(len(gt_boxes)),
        "count_error": float(abs(len(sorted_preds) - len(gt_boxes))),
    }


def stream_sample_plate(
    dataset_id: str = "andreassag/agar_leodriesch_singleclass",
    split: str = "validation",
    index: int = 0,
) -> dict[str, Any]:
    """Stream one sample Petri dish record from Hugging Face for quick inference demos.

    Requires the optional ``datasets`` package; raises a clear ``ImportError`` if absent.
    """
    try:
        datasets_mod = importlib.import_module("datasets")
        load_dataset = datasets_mod.load_dataset
    except ImportError as exc:
        raise ImportError(
            "Streaming sample plates requires the optional 'datasets' package (`pip install datasets`)."
        ) from exc

    stream = load_dataset(dataset_id, split=split, streaming=True)
    for idx, sample in enumerate(stream):
        if idx == index:
            return dict(sample)
    raise IndexError(f"Sample index {index} out of range for split '{split}' of '{dataset_id}'.")


def plot_plate_detections(
    image: np.ndarray | Image.Image,
    detections: Sequence[Detection],
    *,
    gt_boxes: Sequence[tuple[float, float, float, float]] | None = None,
    output_path: str | Path | None = None,
    show_labels: bool = False,
) -> Image.Image:
    """Render predictions (circles/polygons/boxes) and optional GT boxes onto a dish image.

    Args:
        image: Source RGB dish image.
        detections: Model detections to visualize in bright green/cyan.
        gt_boxes: Optional ground-truth boxes drawn in amber dashed/thin outlines.
        output_path: Optional file path to save the rendered PNG/JPG image.
        show_labels: Whether to draw prediction confidence values.

    Returns:
        Annotated ``PIL.Image.Image``.
    """
    if isinstance(image, Image.Image):
        canvas = image.convert("RGB").copy()
    else:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            scale = 255.0 if arr.size and float(np.max(arr)) <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)
        canvas = Image.fromarray(arr).convert("RGB")

    draw = ImageDraw.Draw(canvas)
    if gt_boxes:
        for gt_box in gt_boxes:
            draw.rectangle(gt_box, outline=(255, 193, 7), width=2)

    for det in detections:
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

    if output_path is not None:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest)

    return canvas
