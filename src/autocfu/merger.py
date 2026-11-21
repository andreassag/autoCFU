"""Detection data structures, radial/polygon geometry, and cross-tile merging algorithms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

VALID_MATCH_METRICS: tuple[str, ...] = ("center_dist", "ios", "iou")


def reconstruct_star_polygon(cx: float, cy: float, rays: Sequence[float]) -> list[tuple[float, float]]:
    """Reconstruct 2D star-convex polygon vertices from centroid and equiangular rays.

    Angles follow the standard StarDist radial convention ``theta_k = 2 * pi * k / N``
    with ``x_k = cx + r_k * cos(theta_k)`` and ``y_k = cy + r_k * sin(theta_k)``.

    Args:
        cx: Center x-coordinate in pixels.
        cy: Center y-coordinate in pixels.
        rays: Sequence of radial distances along each equiangular ray.

    Returns:
        List of ``(x, y)`` polygon vertex coordinates.

    Raises:
        ValueError: If ``rays`` is empty.
    """
    num_rays = len(rays)
    if num_rays == 0:
        raise ValueError("Cannot reconstruct a star-convex polygon from an empty ray sequence.")

    two_pi_over_n = (2.0 * math.pi) / float(num_rays)
    vertices: list[tuple[float, float]] = []
    for idx, radius in enumerate(rays):
        theta = float(idx) * two_pi_over_n
        vx = float(cx) + float(radius) * math.cos(theta)
        vy = float(cy) + float(radius) * math.sin(theta)
        vertices.append((vx, vy))
    return vertices


def _polygon_bounds(polygon: Sequence[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class Detection:
    """Single detected or segmented microbial colony in pixel coordinates.

    Supports bounding-box detections, CenterNet radial circles ``(cx, cy, r)``,
    and StarDist star-convex polygons ``(cx, cy, rays, polygon)`` under one
    unified structure.

    Attributes:
        bbox: Bounding box ``(xmin, ymin, xmax, ymax)`` in pixels.
        score: Model confidence score in ``[0.0, 1.0]``.
        class_id: Integer foreground class index.
        label: Human-readable class name.
        center_xy: Optical or geometric colony center ``(cx, cy)`` in pixels.
        radius_px: Radial colony size in pixels (mean ray length for StarDist).
        rays: Radial distances along ``N`` equiangular StarDist rays, if available.
        polygon: Polygon vertices ``[(x0, y0), ...]``, if available.
    """

    bbox: tuple[float, float, float, float] | None = None
    score: float = 0.0
    class_id: int = 0
    label: str = "colony"
    center_xy: tuple[float, float] | None = None
    radius_px: float | None = None
    rays: list[float] | None = None
    polygon: list[tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        self.score = float(self.score)
        self.class_id = int(self.class_id)

        if self.center_xy is not None:
            self.center_xy = (float(self.center_xy[0]), float(self.center_xy[1]))
        if self.radius_px is not None:
            self.radius_px = float(self.radius_px)
        if self.rays is not None:
            self.rays = [float(r) for r in self.rays]
        if self.polygon is not None:
            self.polygon = [(float(px), float(py)) for px, py in self.polygon]

        if self.polygon is None and self.center_xy is not None and self.rays:
            self.polygon = reconstruct_star_polygon(self.center_xy[0], self.center_xy[1], self.rays)

        if self.bbox is None:
            if self.polygon:
                self.bbox = _polygon_bounds(self.polygon)
            elif self.center_xy is not None and self.radius_px is not None:
                cx, cy = self.center_xy
                r = max(0.0, self.radius_px)
                self.bbox = (cx - r, cy - r, cx + r, cy + r)
            else:
                raise ValueError("Detection requires 'bbox', 'polygon', or 'center_xy' with 'radius_px'/'rays'.")
        else:
            self.bbox = (
                float(self.bbox[0]),
                float(self.bbox[1]),
                float(self.bbox[2]),
                float(self.bbox[3]),
            )

        if self.center_xy is None:
            x1, y1, x2, y2 = self.bbox
            self.center_xy = (0.5 * (x1 + x2), 0.5 * (y1 + y2))

        if self.radius_px is None:
            if self.rays:
                self.radius_px = sum(self.rays) / float(len(self.rays))
            else:
                self.radius_px = 0.25 * (self.width + self.height)

    @property
    def width(self) -> float:
        """Bounding-box width in pixels."""
        bbox = self.bbox
        if bbox is None:
            return 0.0
        return max(0.0, bbox[2] - bbox[0])

    @property
    def height(self) -> float:
        """Bounding-box height in pixels."""
        bbox = self.bbox
        if bbox is None:
            return 0.0
        return max(0.0, bbox[3] - bbox[1])

    @property
    def area(self) -> float:
        """Bounding-box area in square pixels."""
        return self.width * self.height

    def translated(self, dx: float, dy: float) -> Detection:
        """Return a copy of this detection translated by ``(dx, dy)`` pixels.

        Args:
            dx: Horizontal offset in pixels.
            dy: Vertical offset in pixels.

        Returns:
            New :class:`Detection` in the translated coordinate frame.
        """
        bbox = self.bbox
        center = self.center_xy
        if bbox is None or center is None:
            raise ValueError("Detection must define both bbox and center_xy before translation.")
        x1, y1, x2, y2 = bbox
        cx, cy = center
        new_polygon = [(px + dx, py + dy) for px, py in self.polygon] if self.polygon is not None else None
        new_rays = list(self.rays) if self.rays is not None else None
        return Detection(
            bbox=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
            score=self.score,
            class_id=self.class_id,
            label=self.label,
            center_xy=(cx + dx, cy + dy),
            radius_px=self.radius_px,
            rays=new_rays,
            polygon=new_polygon,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize detection fields into a JSON-compatible dictionary."""
        bbox = self.bbox
        center = self.center_xy
        if bbox is None or center is None:
            raise ValueError("Detection must define both bbox and center_xy for serialization.")
        payload: dict[str, Any] = {
            "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
            "center_xy": [center[0], center[1]],
            "radius_px": self.radius_px,
            "score": self.score,
            "class_id": self.class_id,
            "label": self.label,
        }
        if self.rays is not None:
            payload["rays"] = list(self.rays)
        if self.polygon is not None:
            payload["polygon"] = [[px, py] for px, py in self.polygon]
        return payload


def _box_intersection_area(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def calculate_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Union (IoU) between two ``(xmin, ymin, xmax, ymax)`` boxes."""
    inter = _box_intersection_area(box_a, box_b)
    if inter <= 0.0:
        return 0.0
    union = _box_area(box_a) + _box_area(box_b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def calculate_ios(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Smaller Area (IoS) between two boxes."""
    inter = _box_intersection_area(box_a, box_b)
    if inter <= 0.0:
        return 0.0
    smaller = min(_box_area(box_a), _box_area(box_b))
    if smaller <= 0.0:
        return 0.0
    return inter / smaller


def calculate_center_distance_ratio(det_a: Detection, det_b: Detection) -> float:
    """Compute Euclidean center distance normalized by the smaller detection radius.

    For CenterNet and StarDist colonies across overlapping tile seams, normalizing
    ``dist(center_a, center_b)`` by ``min(r_a, r_b)`` yields a scale-invariant
    proximity metric where smaller colonies nested near larger ones are only merged
    when their centers fall inside the smaller colony radius.

    Args:
        det_a: First detection.
        det_b: Second detection.

    Returns:
        Dimensionless distance ratio ``d / min(r_a, r_b)``. Returns ``0.0`` when
        centers coincide and ``inf`` when centers differ with zero minimum radius.
    """
    center_a = det_a.center_xy
    center_b = det_b.center_xy
    if center_a is None or center_b is None:
        raise ValueError("Both detections must define center_xy to compute center distance ratio.")
    dx = center_a[0] - center_b[0]
    dy = center_a[1] - center_b[1]
    dist = math.hypot(dx, dy)
    r_a = det_a.radius_px if det_a.radius_px is not None else 0.0
    r_b = det_b.radius_px if det_b.radius_px is not None else 0.0
    min_r = min(r_a, r_b)
    if min_r <= 0.0:
        return 0.0 if dist == 0.0 else math.inf
    return dist / min_r


def detections_match(
    det_a: Detection,
    det_b: Detection,
    match_threshold: float,
    match_metric: str = "center_dist",
) -> bool:
    """Return ``True`` if two detections match according to ``match_metric``."""
    bbox_a = det_a.bbox
    bbox_b = det_b.bbox
    if bbox_a is None or bbox_b is None:
        raise ValueError("Both detections must define bbox to evaluate overlap.")
    metric = match_metric.strip().lower()
    if metric == "center_dist":
        return calculate_center_distance_ratio(det_a, det_b) <= match_threshold
    if metric == "ios":
        return calculate_ios(bbox_a, bbox_b) >= match_threshold
    if metric == "iou":
        return calculate_iou(bbox_a, bbox_b) >= match_threshold
    raise ValueError(f"Unsupported match_metric '{match_metric}'. Expected one of {VALID_MATCH_METRICS}.")


def _validate_match_metric(match_metric: str) -> str:
    metric = match_metric.strip().lower()
    if metric not in VALID_MATCH_METRICS:
        raise ValueError(f"Unsupported match_metric '{match_metric}'. Expected one of {VALID_MATCH_METRICS}.")
    return metric


def _merge_cluster(cluster: Sequence[Detection]) -> Detection:
    """Combine a matched cluster via confidence-weighted centroid, radius, and ray averaging."""
    anchor = cluster[0]
    if len(cluster) == 1:
        return anchor

    raw_weights = [max(det.score, 0.0) for det in cluster]
    weight_sum = sum(raw_weights)
    if weight_sum <= 0.0:
        weights = [1.0 for _ in cluster]
        weight_sum = float(len(cluster))
    else:
        weights = raw_weights

    cx = sum(w * (det.center_xy[0] if det.center_xy is not None else 0.0) for w, det in zip(weights, cluster))
    cy = sum(w * (det.center_xy[1] if det.center_xy is not None else 0.0) for w, det in zip(weights, cluster))
    cx /= weight_sum
    cy /= weight_sum

    radius = sum(w * (det.radius_px if det.radius_px is not None else 0.0) for w, det in zip(weights, cluster))
    radius /= weight_sum

    # Average equiangular radial vectors when all cluster members are StarDist detections
    first_rays = anchor.rays
    merged_rays: list[float] | None = None
    merged_polygon: list[tuple[float, float]] | None = None
    cluster_rays = [
        (w, det.rays)
        for w, det in zip(weights, cluster)
        if det.rays is not None and first_rays is not None and len(det.rays) == len(first_rays)
    ]
    if first_rays is not None and len(cluster_rays) == len(cluster):
        num_rays = len(first_rays)
        merged_rays = [0.0] * num_rays
        for w, rays in cluster_rays:
            for k in range(num_rays):
                merged_rays[k] += w * rays[k]
        merged_rays = [r / weight_sum for r in merged_rays]
        merged_polygon = reconstruct_star_polygon(cx, cy, merged_rays)
        merged_bbox = _polygon_bounds(merged_polygon)
        radius = sum(merged_rays) / float(num_rays)
    else:
        bx1 = sum(w * det.bbox[0] for w, det in zip(weights, cluster) if det.bbox is not None) / weight_sum
        by1 = sum(w * det.bbox[1] for w, det in zip(weights, cluster) if det.bbox is not None) / weight_sum
        bx2 = sum(w * det.bbox[2] for w, det in zip(weights, cluster) if det.bbox is not None) / weight_sum
        by2 = sum(w * det.bbox[3] for w, det in zip(weights, cluster) if det.bbox is not None) / weight_sum
        merged_bbox = (bx1, by1, bx2, by2)

    return Detection(
        bbox=merged_bbox,
        score=max(det.score for det in cluster),
        class_id=anchor.class_id,
        label=anchor.label,
        center_xy=(cx, cy),
        radius_px=radius,
        rays=merged_rays,
        polygon=merged_polygon,
    )


class GreedyNMM:
    """Greedy Non-Maximum Merging for deduplicating colonies across tile seams.

    Processes detections in descending score order, clusters overlapping matches
    of the same class, and merges each cluster using confidence-weighted averaging
    of centroids, radii, bounding boxes, and StarDist rays.
    """

    def __init__(self, match_threshold: float = 0.50, match_metric: str = "center_dist") -> None:
        self.match_threshold = float(match_threshold)
        self.match_metric = _validate_match_metric(match_metric)

    def merge(self, detections: Sequence[Detection]) -> list[Detection]:
        """Merge overlapping detections and return deduplicated predictions."""
        if not detections:
            return []

        ordered = [det for _, det in sorted(enumerate(detections), key=lambda pair: (-pair[1].score, pair[0]))]
        consumed = [False] * len(ordered)
        merged: list[Detection] = []

        for i, anchor in enumerate(ordered):
            if consumed[i]:
                continue
            consumed[i] = True
            cluster = [anchor]
            for j in range(i + 1, len(ordered)):
                if consumed[j]:
                    continue
                cand = ordered[j]
                if cand.class_id != anchor.class_id:
                    continue
                if detections_match(anchor, cand, self.match_threshold, self.match_metric):
                    consumed[j] = True
                    cluster.append(cand)
            merged.append(_merge_cluster(cluster))

        return merged


class NMS:
    """Standard Non-Maximum Suppression supporting ``center_dist``, ``ios``, and ``iou``."""

    def __init__(self, match_threshold: float = 0.45, match_metric: str = "center_dist") -> None:
        self.match_threshold = float(match_threshold)
        self.match_metric = _validate_match_metric(match_metric)

    def merge(self, detections: Sequence[Detection]) -> list[Detection]:
        """Suppress overlapping lower-confidence detections."""
        if not detections:
            return []

        ordered = [det for _, det in sorted(enumerate(detections), key=lambda pair: (-pair[1].score, pair[0]))]
        suppressed = [False] * len(ordered)
        kept: list[Detection] = []

        for i, anchor in enumerate(ordered):
            if suppressed[i]:
                continue
            kept.append(anchor)
            for j in range(i + 1, len(ordered)):
                if suppressed[j]:
                    continue
                cand = ordered[j]
                if cand.class_id != anchor.class_id:
                    continue
                if detections_match(anchor, cand, self.match_threshold, self.match_metric):
                    suppressed[j] = True

        return kept


def get_merger(
    merger_type: str = "greedy_nmm",
    match_threshold: float = 0.50,
    match_metric: str = "center_dist",
) -> GreedyNMM | NMS | None:
    """Instantiate a merger strategy by name.

    Args:
        merger_type: One of ``"greedy_nmm"``, ``"nms"``, or ``"none"``.
        match_threshold: Threshold for considering two detections overlapping.
        match_metric: Overlap metric (``"center_dist"``, ``"ios"``, or ``"iou"``).

    Returns:
        Configured merger instance, or ``None`` when ``merger_type == "none"``.

    Raises:
        ValueError: If ``merger_type`` or ``match_metric`` is invalid.
    """
    kind = merger_type.strip().lower()
    if kind == "greedy_nmm":
        return GreedyNMM(match_threshold=match_threshold, match_metric=match_metric)
    if kind == "nms":
        return NMS(match_threshold=match_threshold, match_metric=match_metric)
    if kind == "none":
        _validate_match_metric(match_metric)
        return None
    raise ValueError(f"Unsupported merger_type '{merger_type}'. Expected 'greedy_nmm', 'nms', or 'none'.")
