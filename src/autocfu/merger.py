"""Bounding box clustering and post-processing mergers (GreedyNMM and NMS)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class Detection:
    """Represents a single object detection in absolute coordinates."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    score: float
    class_id: int = 0
    label: str = "colony"

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        """X coordinate of bounding box center / centroid."""
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        """Y coordinate of bounding box center / centroid."""
        return (self.y1 + self.y2) / 2.0

    @property
    def center(self) -> tuple[float, float]:
        """(cx, cy) tuple representing the colony centroid."""
        return (self.cx, self.cy)

    @property
    def radius(self) -> float:
        """Estimated colony radius based on average bounding box dimension."""
        return (self.width + self.height) / 4.0

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(float(c), 2) for c in self.bbox],
            "center": [round(float(self.cx), 2), round(float(self.cy), 2)],
            "score": round(float(self.score), 4),
            "class_id": int(self.class_id),
            "label": str(self.label),
        }


def calculate_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Calculate Intersection-over-Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    if intersection_area <= 0.0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def calculate_ios(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Calculate Intersection-over-Smaller (IoS) of two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    if intersection_area <= 0.0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    min_area = min(area_a, area_b)

    if min_area <= 0.0:
        return 0.0

    return intersection_area / min_area


class GreedyNMM:
    """Greedy Non-Maximum Merging (GreedyNMM).

    Clusters overlapping detections exceeding an IoU/IoS threshold and computes
    confidence-weighted average coordinates for merged clusters.
    """

    def __init__(
        self,
        iou_threshold: float = 0.50,
        class_agnostic: bool = False,
        match_metric: str = "iou",
        *,
        match_threshold: float | None = None,
    ) -> None:
        if match_threshold is not None:
            iou_threshold = match_threshold
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        self.iou_threshold = iou_threshold
        self.class_agnostic = class_agnostic
        self.match_metric = match_metric.lower()
        if self.match_metric not in ("iou", "ios"):
            raise ValueError(f"Unknown match_metric: {match_metric}. Supported: 'iou', 'ios'.")
        self._metric_fn = calculate_ios if self.match_metric == "ios" else calculate_iou

    def merge(self, detections: list[Detection]) -> list[Detection]:
        """Merge overlapping detections into consolidated detections."""
        if not detections:
            return []

        if self.class_agnostic:
            return self._merge_class_group(detections)

        # Group by class_id
        grouped: dict[int, list[Detection]] = defaultdict(list)
        for det in detections:
            grouped[det.class_id].append(det)

        merged_results: list[Detection] = []
        for class_id, class_dets in grouped.items():
            merged_results.extend(self._merge_class_group(class_dets))

        # Sort all merged detections by score descending
        merged_results.sort(key=lambda d: d.score, reverse=True)
        return merged_results

    def _merge_class_group(self, detections: list[Detection]) -> list[Detection]:
        # Sort descending by score
        candidates = sorted(detections, key=lambda d: d.score, reverse=True)
        merged: list[Detection] = []

        while candidates:
            best = candidates.pop(0)
            cluster = [best]
            remaining: list[Detection] = []

            for det in candidates:
                match_val = self._metric_fn(best.bbox, det.bbox)
                if match_val >= self.iou_threshold:
                    cluster.append(det)
                else:
                    remaining.append(det)

            candidates = remaining

            # Compute weighted coordinates for the cluster
            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                total_weight = sum(d.score for d in cluster)
                if total_weight > 0:
                    merged_x1 = sum(d.x1 * d.score for d in cluster) / total_weight
                    merged_y1 = sum(d.y1 * d.score for d in cluster) / total_weight
                    merged_x2 = sum(d.x2 * d.score for d in cluster) / total_weight
                    merged_y2 = sum(d.y2 * d.score for d in cluster) / total_weight
                else:
                    merged_x1 = sum(d.x1 for d in cluster) / len(cluster)
                    merged_y1 = sum(d.y1 for d in cluster) / len(cluster)
                    merged_x2 = sum(d.x2 for d in cluster) / len(cluster)
                    merged_y2 = sum(d.y2 for d in cluster) / len(cluster)

                max_score = max(d.score for d in cluster)
                merged.append(
                    Detection(
                        bbox=(merged_x1, merged_y1, merged_x2, merged_y2),
                        score=max_score,
                        class_id=best.class_id,
                        label=best.label,
                    )
                )

        return merged


class NMS:
    """Standard Non-Maximum Suppression (NMS).

    Suppresses overlapping candidate boxes with lower scores without coordinate averaging.
    """

    def __init__(
        self,
        iou_threshold: float = 0.50,
        class_agnostic: bool = False,
        match_metric: str = "iou",
        *,
        match_threshold: float | None = None,
    ) -> None:
        if match_threshold is not None:
            iou_threshold = match_threshold
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        self.iou_threshold = iou_threshold
        self.class_agnostic = class_agnostic
        self.match_metric = match_metric.lower()
        if self.match_metric not in ("iou", "ios"):
            raise ValueError(f"Unknown match_metric: {match_metric}. Supported: 'iou', 'ios'.")
        self._metric_fn = calculate_ios if self.match_metric == "ios" else calculate_iou

    def merge(self, detections: list[Detection]) -> list[Detection]:
        """Apply standard NMS suppression."""
        if not detections:
            return []

        if self.class_agnostic:
            return self._suppress_group(detections)

        grouped: dict[int, list[Detection]] = defaultdict(list)
        for det in detections:
            grouped[det.class_id].append(det)

        survivors: list[Detection] = []
        for class_id, class_dets in grouped.items():
            survivors.extend(self._suppress_group(class_dets))

        survivors.sort(key=lambda d: d.score, reverse=True)
        return survivors

    def _suppress_group(self, detections: list[Detection]) -> list[Detection]:
        candidates = sorted(detections, key=lambda d: d.score, reverse=True)
        kept: list[Detection] = []

        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            candidates = [det for det in candidates if self._metric_fn(best.bbox, det.bbox) < self.iou_threshold]

        return kept


def get_merger(
    merger_type: str = "greedy_nmm",
    iou_threshold: float = 0.50,
    match_metric: str = "iou",
    *,
    match_threshold: float | None = None,
) -> GreedyNMM | NMS:
    """Factory creating the specified post-processing merger."""
    if match_threshold is not None:
        iou_threshold = match_threshold
    m_type = merger_type.strip().lower()
    if m_type in ("greedy_nmm", "greedynmm", "nmm"):
        return GreedyNMM(iou_threshold=iou_threshold, match_metric=match_metric)
    elif m_type in ("nms", "standard_nms"):
        return NMS(iou_threshold=iou_threshold, match_metric=match_metric)
    else:
        raise ValueError(f"Unknown merger_type: {merger_type}. Supported: 'greedy_nmm', 'nms'.")
