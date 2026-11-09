"""Utility functions for dataset inspection, ground-truth extraction, and visualization."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from autocfu.merger import calculate_iou

logger = logging.getLogger(__name__)

KNOWN_COLONY_LABELS: set[str | int] = {
    "colony",
    "e.coli",
    "e. coli",
    "s.aureus",
    "s. aureus",
    "b.subtilis",
    "b. subtilis",
    "c.albicans",
    "c. albicans",
    "p.aeruginosa",
    "p. aeruginosa",
    "contamination",
    "0",
    "1",
    "2",
    "3",
    "4",
    "6",
    0,
    1,
    2,
    3,
    4,
    6,
}
IGNORED_CATEGORIES: set[str | int] = {"Defect", "defect", "agar_bubble", 5, "5"}


def is_colony_label(label: str | int) -> bool:
    """Determine whether a category label corresponds to a colony or contamination.

    Args:
        label: Raw category label or class ID from plate annotations.

    Returns:
        True if label represents a colony target, False for defects and artifacts.
    """
    clean_label = str(label).strip().lower()
    if any(str(ign).lower() == clean_label or str(ign).lower() in clean_label for ign in IGNORED_CATEGORIES):
        return False
    clean_no_spaces = clean_label.replace(" ", "")
    return (
        clean_label in KNOWN_COLONY_LABELS
        or clean_no_spaces in KNOWN_COLONY_LABELS
        or "colony" in clean_label
        or "contamination" in clean_label
        or (isinstance(label, int) and label in KNOWN_COLONY_LABELS)
    )


def is_valid_highres_plate(sample: dict[str, Any], min_dim: int = 3000) -> bool:
    """Validate that an AGAR dataset sample is a full-resolution countable Petri dish.

    Args:
        sample: Raw sample dictionary from AGAR dataset.
        min_dim: Minimum acceptable image dimension in pixels (default: 3000).

    Returns:
        True if the image is a high-resolution, countable Petri dish.
    """
    img = sample.get("image")
    if img is None:
        return False

    if hasattr(img, "size"):
        w, h = img.size
    elif hasattr(img, "shape"):
        h, w = img.shape[:2]
    else:
        return False

    if w < min_dim or h < min_dim or (w == 2048 and h == 2048):
        return False

    col_num = sample.get("colonies_number")
    if col_num is not None:
        try:
            if int(col_num) == -1:
                return False
        except (ValueError, TypeError):
            pass

    if sample.get("image_type") in (2, "uncountable"):
        return False

    return True


def extract_gt_colony_boxes(sample: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    """Extract ground-truth colony bounding boxes [x1, y1, x2, y2] in plate coordinates.

    Handles both standard AGAR dictionary format ('objects' with 'bbox' and 'category')
    and flat formats ('bboxes'/'boxes' and 'labels'/'categories').

    Args:
        sample: Dictionary containing plate annotations.

    Returns:
        List of bounding boxes in [x1, y1, x2, y2] float format.
    """
    gt_boxes: list[tuple[float, float, float, float]] = []

    # AGAR dataset format: sample['objects'] -> {'bbox': [[x, y, w, h], ...], 'category': [...]}
    objects = sample.get("objects") or {}
    raw_bboxes = (objects.get("bbox") or []) if isinstance(objects, dict) else []
    raw_categories = (objects.get("category") or objects.get("label") or []) if isinstance(objects, dict) else []
    if raw_bboxes:
        for idx, bbox in enumerate(raw_bboxes):
            cat = raw_categories[idx] if idx < len(raw_categories) else 0
            if not is_colony_label(cat):
                continue
            if len(bbox) == 4:
                bx, by, bw, bh = bbox
                if bw > 0 and bh > 0:
                    gt_boxes.append((float(bx), float(by), float(bx + bw), float(by + bh)))
        return gt_boxes

    # Fallback for flat schemas: sample['bboxes'] / sample['boxes']
    raw_boxes = sample.get("bboxes") or sample.get("boxes") or []
    labels = sample.get("labels") or sample.get("categories") or []
    for idx, b in enumerate(raw_boxes):
        lbl = labels[idx] if idx < len(labels) else 0
        if not is_colony_label(lbl):
            continue
        if len(b) == 4:
            gt_boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))

    return gt_boxes


def compute_detection_metrics(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    scores: Sequence[float] | None = None,
    iou_threshold: float = 0.50,
) -> dict[str, float]:
    """Compute greedy IoU-matched detection metrics at a specified IoU threshold.

    Sorts predictions by descending confidence score when provided, matching each prediction
    greedily to the ground truth box with the highest IoU >= iou_threshold.

    Args:
        pred_boxes: Sequence of predicted boxes [x1, y1, x2, y2].
        gt_boxes: Sequence of ground-truth boxes [x1, y1, x2, y2].
        scores: Optional sequence of confidence scores corresponding to pred_boxes.
        iou_threshold: Intersection-over-Union threshold for a true positive match.

    Returns:
        Dictionary with keys 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'abs_err', 'count_acc'.
    """
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)
    abs_err = abs(num_pred - num_gt)
    count_acc = max(0.0, 1.0 - (abs_err / max(1, num_gt)))

    if num_gt == 0 and num_pred == 0:
        return {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "abs_err": 0.0,
            "count_acc": 1.0,
        }

    if num_gt == 0:
        return {
            "tp": 0.0,
            "fp": float(num_pred),
            "fn": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "abs_err": float(abs_err),
            "count_acc": 0.0,
        }

    if num_pred == 0:
        return {
            "tp": 0.0,
            "fp": 0.0,
            "fn": float(num_gt),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "abs_err": float(abs_err),
            "count_acc": round(count_acc, 4),
        }

    if scores is not None and len(scores) == len(pred_boxes):
        sorted_indices = sorted(range(len(pred_boxes)), key=lambda k: scores[k], reverse=True)
        sorted_preds = [pred_boxes[k] for k in sorted_indices]
    else:
        sorted_preds = list(pred_boxes)

    matched_gt: set[int] = set()
    tp = 0
    for p in sorted_preds:
        best_iou, best_idx = 0.0, -1
        for idx, g in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = calculate_iou(p, g)
            if iou >= iou_threshold and iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1

    fp = num_pred - tp
    fn = num_gt - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-6, precision + recall)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "abs_err": float(abs_err),
        "count_acc": round(count_acc, 4),
    }


def stream_sample_plate(
    subset: str = "bright",
    index: int = 0,
    dataset_id: str = "leodriesch/agar",
    split: str = "validation",
    hf_token: str | None = None,
    min_dim: int = 3000,
) -> tuple[np.ndarray, str]:
    """Stream a high-resolution plate record from Hugging Face Hub.

    Args:
        subset: Dataset subset or illumination ('bright' or 'dark').
        index: Zero-based index of valid plate to select.
        dataset_id: Hugging Face dataset identifier.
        split: Dataset split ('train', 'validation', or 'test').
        hf_token: Optional Hugging Face authentication token.
        min_dim: Minimum image dimension for validity filtering.

    Returns:
        Tuple of (image_rgb_array, description_title).

    Raises:
        ValueError: If index is negative.
        ImportError: If the 'datasets' package is not installed.
        RuntimeError: If no valid plate is found at the given index.
    """
    if index < 0:
        raise ValueError(f"Index must be non-negative, got {index}.")

    import importlib

    try:
        load_dataset = importlib.import_module("datasets").load_dataset
    except ImportError as err:
        raise ImportError(
            "The 'datasets' package is required to stream plates from Hugging Face Hub. "
            "Install it via 'pip install datasets'."
        ) from err

    import os

    token = hf_token or os.environ.get("HF_TOKEN") or None
    ds = load_dataset(dataset_id, subset, split=split, streaming=True, token=token)
    valid_idx = 0
    for item in ds:
        if is_valid_highres_plate(item, min_dim=min_dim):
            if valid_idx >= index:
                image = item["image"]
                rgb = np.asarray(image.convert("RGB"))
                img_id = item.get("image_id", f"{subset}_{valid_idx}")
                sub_clean = subset.lower().removesuffix("field")
                title = f"{sub_clean.capitalize()}field Plate (ID: {img_id}, Size: {rgb.shape[1]}x{rgb.shape[0]} px)"
                return rgb, title
            valid_idx += 1

    raise RuntimeError(f"Could not find a valid high-resolution plate in '{subset}' split at index {index}.")


def plot_plate_detections(
    image_rgb: np.ndarray,
    detections: Sequence[Any],
    title: str,
    duration: float | None = None,
    crop_size: int = 250,
) -> list[tuple[int, int, int, int, float]]:
    """Render full-plate annotated image and a 3x3 grid of native-resolution zoom crops.

    Args:
        image_rgb: Full plate image as RGB NumPy array.
        detections: List of detections (Detection instances, SAHI ObjectPrediction, or tuples/lists).
        title: Title string for the full dish visualization.
        duration: Optional inference execution time in seconds.
        crop_size: Pixel dimension of square zoom crops (default: 250).

    Returns:
        List of tuples (cx, cy, width, height, score) for each detection.

    Raises:
        ImportError: If 'opencv-python' or 'matplotlib' is not installed.
    """
    import importlib

    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as err:
        raise ImportError(
            "The 'opencv-python' package is required for plot_plate_detections. "
            "Install it via 'pip install opencv-python-headless'."
        ) from err

    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as err:
        raise ImportError(
            "The 'matplotlib' package is required for plot_plate_detections. Install it via 'pip install matplotlib'."
        ) from err

    annotated = image_rgb.copy() if hasattr(image_rgb, "copy") else image_rgb
    plate_h, plate_w = image_rgb.shape[:2]

    colony_coords: list[tuple[int, int, int, int, float]] = []
    for det in detections:
        if hasattr(det, "bbox") and isinstance(det.bbox, (tuple, list)):
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
            score_val = float(det.score)
        elif hasattr(det, "bbox"):
            b = det.bbox
            x1, y1, x2, y2 = int(round(b.minx)), int(round(b.miny)), int(round(b.maxx)), int(round(b.maxy))
            score_val = float(det.score.value)
        else:
            x1, y1, x2, y2 = [int(round(v)) for v in det[:4]]
            score_val = float(det[4]) if len(det) > 4 else 1.0
        cx, cy = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
        colony_coords.append((cx, cy, x2 - x1, y2 - y1, score_val))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

    # 1. Full-plate render
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(annotated)
    dur_str = f" in {duration:.2f}s" if duration is not None else ""
    ax.set_title(f"{title}\nDetections: {len(detections)} Colonies{dur_str}", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    # 2. 3x3 Zoom panels
    if colony_coords:
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        num_samples = min(9, len(colony_coords))
        sample_indices = np.linspace(0, len(colony_coords) - 1, num_samples).astype(int)
        for i, ax in enumerate(axes.ravel()):
            if i < len(sample_indices):
                idx = sample_indices[i]
                cx, cy, cw, ch, sc = colony_coords[idx]
                x_min = max(0, min(cx - crop_size // 2, max(0, plate_w - crop_size)))
                y_min = max(0, min(cy - crop_size // 2, max(0, plate_h - crop_size)))
                x_max = min(plate_w, x_min + crop_size)
                y_max = min(plate_h, y_min + crop_size)
                crop = annotated[y_min:y_max, x_min:x_max]
                ax.imshow(crop)
                ax.set_title(f"Colony #{idx + 1} ({sc:.2f}) | {cw}x{ch} px", fontsize=9)
            ax.axis("off")
        plt.suptitle("Native-Resolution Zoom Inspection (3x3 Panels)", fontsize=13)
        plt.tight_layout()
        plt.show()

    return colony_coords


__all__ = [
    "IGNORED_CATEGORIES",
    "KNOWN_COLONY_LABELS",
    "compute_detection_metrics",
    "extract_gt_colony_boxes",
    "is_colony_label",
    "is_valid_highres_plate",
    "plot_plate_detections",
    "stream_sample_plate",
]
