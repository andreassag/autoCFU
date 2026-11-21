"""ONNX Runtime model runners for CenterNet, 2D StarDist, and Countability inference."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from autocfu.merger import NMS, Detection, GreedyNMM, reconstruct_star_polygon

COUNTABILITY_LABELS: tuple[str, ...] = ("empty", "countable", "uncountable")
VALID_EXPLICIT_MODEL_TYPES: tuple[str, ...] = ("centernet", "stardist", "countability")


@dataclass
class CountabilityResult:
    """Prediction emitted by a Plate/Section Countability ONNX model.

    Attributes:
        probs: Class probabilities ``(p_empty, p_countable, p_uncountable)``.
        logits: Raw model logits ``(l_empty, l_countable, l_uncountable)``.
        pred_class_id: Argmax class index in ``{0: empty, 1: countable, 2: uncountable}``.
        pred_label: Class name corresponding to ``pred_class_id``.
        is_countable: Binary countability decision (``p_empty + p_countable >= 1 - threshold``).
        countable_probability: Combined probability of being ``empty`` or ``countable``.
        uncountable_probability: Probability of class ``uncountable``.
        region_bbox: Optional ``(xmin, ymin, xmax, ymax)`` pixel region for section predictions.
    """

    probs: tuple[float, float, float]
    logits: tuple[float, float, float]
    pred_class_id: int
    pred_label: str
    is_countable: bool
    countable_probability: float
    uncountable_probability: float
    region_bbox: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize countability result to a JSON-compatible dictionary."""
        payload: dict[str, Any] = {
            "probs": {
                "empty": self.probs[0],
                "countable": self.probs[1],
                "uncountable": self.probs[2],
            },
            "logits": [self.logits[0], self.logits[1], self.logits[2]],
            "pred_class_id": self.pred_class_id,
            "pred_label": self.pred_label,
            "is_countable": self.is_countable,
            "countable_probability": self.countable_probability,
            "uncountable_probability": self.uncountable_probability,
        }
        if self.region_bbox is not None:
            payload["region_bbox"] = list(self.region_bbox)
        return payload


def _safe_unbatch(tensor: np.ndarray) -> np.ndarray:
    """Remove leading batch dimension when batch size is 1."""
    arr = np.asarray(tensor)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _find_output_tensor(
    output_names: Sequence[str],
    outputs: Sequence[np.ndarray],
    candidates: Sequence[str],
) -> np.ndarray | None:
    """Look up an ONNX output tensor by exact or case-insensitive candidate name."""
    name_to_idx: dict[str, int] = {name: idx for idx, name in enumerate(output_names)}
    lower_to_idx: dict[str, int] = {name.lower(): idx for idx, name in enumerate(output_names)}

    for candidate in candidates:
        if candidate in name_to_idx:
            return np.asarray(outputs[name_to_idx[candidate]])
        low = candidate.lower()
        if low in lower_to_idx:
            return np.asarray(outputs[lower_to_idx[low]])
    return None


def _max_pool_3x3(arr: np.ndarray) -> np.ndarray:
    """Apply 3x3 stride-1 max pooling with -inf border padding in pure NumPy."""
    height, width = arr.shape
    padded = np.pad(arr, ((1, 1), (1, 1)), mode="constant", constant_values=-np.inf)
    windows = [padded[dy : dy + height, dx : dx + width] for dy in range(3) for dx in range(3)]
    return np.max(np.stack(windows, axis=0), axis=0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def _softmax_1d(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exp_vals = np.exp(shifted)
    total = float(np.sum(exp_vals))
    if total <= 0.0:
        return np.full_like(logits, 1.0 / float(len(logits)), dtype=np.float32)
    return (exp_vals / total).astype(np.float32)


def _resolve_providers(device: str) -> list[str]:
    req = device.strip().lower()
    available = ort.get_available_providers()
    if req == "cpu":
        return ["CPUExecutionProvider"]
    if req == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"Requested device='cuda', but CUDAExecutionProvider is not available. Available: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if req == "auto":
        providers: list[str] = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers
    raise ValueError(f"Unsupported device '{device}'. Expected 'auto', 'cpu', or 'cuda'.")


def detect_onnx_model_type(
    output_names: Sequence[str],
    metadata: dict[str, str] | None = None,
) -> str:
    """Infer model family (``centernet``, ``stardist``, ``countability``) from ONNX metadata/outputs.

    Args:
        output_names: Ordered ONNX graph output names.
        metadata: Optional ONNX ``custom_metadata_map``.

    Returns:
        Detected model type string.

    Raises:
        ValueError: If model type cannot be uniquely determined from outputs.
    """
    if metadata:
        explicit = metadata.get("autocfu_model_type", "").strip().lower()
        if explicit in VALID_EXPLICIT_MODEL_TYPES:
            return explicit

    lowered = {name.strip().lower() for name in output_names}

    stardist_markers = {"rays", "polygons", "ray_dists", "prob"}
    if lowered & stardist_markers:
        return "stardist"

    centernet_markers = {"radii", "centers", "heatmap", "hm", "radius"}
    if lowered & centernet_markers:
        return "centernet"

    countability_markers = {"logits", "probs", "pred_class", "is_countable"}
    if lowered & countability_markers:
        return "countability"

    raise ValueError(
        f"Unable to auto-detect AutoCFU model type from ONNX outputs {list(output_names)}. "
        f"Specify model_type explicitly ({VALID_EXPLICIT_MODEL_TYPES})."
    )


def _image_to_rgb_array(image: np.ndarray | Image.Image) -> np.ndarray:
    """Convert a PIL Image or NumPy array into a contiguous ``(H, W, 3)`` array."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))

    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3]
    raise ValueError(f"Expected 2D/3D RGB image array, got shape {arr.shape}.")


class BaseONNXModel:
    """Shared ONNX Runtime session setup and image preprocessing base class."""

    model_type: str = "base"

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        input_size: int = 640,
        session: ort.InferenceSession | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.input_size = int(input_size)
        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}.")

        if session is not None:
            self.session = session
        else:
            if not self.model_path.exists():
                raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
            providers = _resolve_providers(device)
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        meta = self.session.get_modelmeta()
        self.metadata: dict[str, str] = dict(meta.custom_metadata_map) if meta and meta.custom_metadata_map else {}

        # Match ONNX input spatial size if statically defined
        if self.session.get_inputs():
            shape = self.session.get_inputs()[0].shape
            if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
                self.input_size = int(shape[2])

    def preprocess(self, image: np.ndarray | Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
        """Resize and normalize an RGB image to NCHW float32 in ``[0.0, 1.0]``.

        Args:
            image: Input RGB image as ``PIL.Image.Image`` or ``np.ndarray``.

        Returns:
            Tuple ``(tensor, (orig_h, orig_w))`` with ``tensor`` shaped ``[1, 3, H, W]``.
        """
        rgb = _image_to_rgb_array(image)
        orig_h, orig_w = int(rgb.shape[0]), int(rgb.shape[1])

        if orig_h != self.input_size or orig_w != self.input_size:
            if np.issubdtype(rgb.dtype, np.floating):
                scaled = np.clip(rgb * 255.0 if rgb.max() <= 1.0 else rgb, 0.0, 255.0).astype(np.uint8)
                pil_img = Image.fromarray(scaled, mode="RGB")
            else:
                pil_img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            resized = pil_img.resize((self.input_size, self.input_size), resample=Image.Resampling.BILINEAR)
            rgb_proc = np.asarray(resized, dtype=np.float32) / 255.0
        else:
            if np.issubdtype(rgb.dtype, np.floating):
                max_val = float(np.max(rgb)) if rgb.size else 0.0
                rgb_proc = rgb.astype(np.float32) / 255.0 if max_val > 1.0 else rgb.astype(np.float32)
            else:
                rgb_proc = rgb.astype(np.float32) / 255.0

        chw = np.transpose(rgb_proc, (2, 0, 1))
        nchw = np.ascontiguousarray(chw[None, :, :, :], dtype=np.float32)
        return nchw, (orig_h, orig_w)

    def _run_session(self, nchw: np.ndarray) -> list[np.ndarray]:
        primary_input = self.input_names[0] if self.input_names else "images"
        feed: dict[str, Any] = {primary_input: nchw}
        if "orig_target_sizes" in self.input_names:
            feed["orig_target_sizes"] = np.array([[self.input_size, self.input_size]], dtype=np.int64)
        outputs = self.session.run(self.output_names, feed)
        return [np.asarray(out) for out in outputs]


class _BaseSpatialColonyONNXModel(BaseONNXModel):
    """Shared filtering and intra-tile suppression helpers for spatial colony models."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        input_size: int = 640,
        top_k: int = 300,
        session: ort.InferenceSession | None = None,
    ) -> None:
        super().__init__(model_path=model_path, device=device, input_size=input_size, session=session)
        self.top_k = int(top_k)

    @staticmethod
    def _filter_and_suppress(
        detections: list[Detection],
        *,
        conf_threshold: float,
        min_radius: float,
        max_radius: float,
        min_box_size: float,
        max_box_size: float,
        max_aspect_ratio: float,
        target_class_id: int | None,
        intra_tile_nms_threshold: float,
        intra_tile_metric: str,
    ) -> list[Detection]:
        filtered: list[Detection] = []
        for det in detections:
            if not math.isfinite(det.score) or det.score < conf_threshold:
                continue
            if target_class_id is not None and det.class_id != target_class_id:
                continue
            radius = det.radius_px if det.radius_px is not None else 0.0
            if radius < min_radius or radius > max_radius:
                continue
            w, h = det.width, det.height
            if w < min_box_size or h < min_box_size:
                continue
            if w > max_box_size or h > max_box_size:
                continue
            min_side = min(w, h)
            if min_side <= 0.0:
                continue
            aspect = max(w / min_side, h / min_side)
            if aspect > max_aspect_ratio:
                continue
            filtered.append(det)

        if not filtered or intra_tile_nms_threshold <= 0.0:
            return filtered

        if intra_tile_metric in ("ios", "iou") and intra_tile_nms_threshold < 1.0:
            return NMS(match_threshold=intra_tile_nms_threshold, match_metric=intra_tile_metric).merge(filtered)
        if intra_tile_metric == "center_dist":
            return GreedyNMM(match_threshold=intra_tile_nms_threshold, match_metric="center_dist").merge(filtered)
        return sorted(filtered, key=lambda det: det.score, reverse=True)


class CenterNetONNXModel(_BaseSpatialColonyONNXModel):
    """ONNX runner for Radial CenterNet (``centernet``) models."""

    model_type: str = "centernet"

    def predict(
        self,
        image: np.ndarray | Image.Image,
        *,
        conf_threshold: float = 0.35,
        min_radius: float = 1.5,
        max_radius: float = 250.0,
        min_box_size: float = 3.0,
        max_box_size: float = 500.0,
        max_aspect_ratio: float = 4.0,
        target_class_id: int | None = 0,
        class_name: str = "colony",
        intra_tile_nms_threshold: float = 0.45,
        intra_tile_metric: str = "ios",
    ) -> list[Detection]:
        """Run Radial CenterNet inference on a single image or tile crop.

        Supports both baked-in decoded ONNX outputs (``boxes``, ``scores``, ``labels``,
        ``centers``, ``radii``) and raw dense head maps (``heatmap``, ``offset``, ``radius``).
        """
        nchw, (orig_h, orig_w) = self.preprocess(image)
        outputs = self._run_session(nchw)
        scale_x = float(orig_w) / float(self.input_size)
        scale_y = float(orig_h) / float(self.input_size)
        scale_r = 0.5 * (scale_x + scale_y)

        scores_t = _find_output_tensor(self.output_names, outputs, ["scores", "pred_scores", "confs"])
        centers_t = _find_output_tensor(self.output_names, outputs, ["centers", "pred_centers"])
        radii_t = _find_output_tensor(self.output_names, outputs, ["radii", "pred_radii"])
        boxes_t = _find_output_tensor(self.output_names, outputs, ["boxes", "pred_boxes", "bboxes"])
        labels_t = _find_output_tensor(self.output_names, outputs, ["labels", "pred_labels", "classes"])

        raw_detections: list[Detection] = []
        if scores_t is not None and (centers_t is not None or boxes_t is not None):
            scores = _safe_unbatch(scores_t).reshape(-1)
            labels = (
                _safe_unbatch(labels_t).reshape(-1).astype(np.int64)
                if labels_t is not None
                else np.zeros_like(scores, dtype=np.int64)
            )
            centers = _safe_unbatch(centers_t).reshape(-1, 2) if centers_t is not None else None
            radii = _safe_unbatch(radii_t).reshape(-1) if radii_t is not None else None
            boxes = _safe_unbatch(boxes_t).reshape(-1, 4) if boxes_t is not None else None

            for idx, score_val in enumerate(scores):
                score_f = float(score_val)
                if score_f < conf_threshold:
                    continue
                cls_id = int(labels[idx])
                if centers is not None and radii is not None:
                    cx = float(centers[idx, 0]) * scale_x
                    cy = float(centers[idx, 1]) * scale_y
                    r = max(0.0, float(radii[idx]) * scale_r)
                    bbox = (
                        (
                            float(boxes[idx, 0]) * scale_x,
                            float(boxes[idx, 1]) * scale_y,
                            float(boxes[idx, 2]) * scale_x,
                            float(boxes[idx, 3]) * scale_y,
                        )
                        if boxes is not None
                        else (cx - r, cy - r, cx + r, cy + r)
                    )
                    raw_detections.append(
                        Detection(
                            bbox=bbox,
                            score=score_f,
                            class_id=cls_id,
                            label=class_name,
                            center_xy=(cx, cy),
                            radius_px=r,
                        )
                    )
                elif boxes is not None:
                    x1 = float(boxes[idx, 0]) * scale_x
                    y1 = float(boxes[idx, 1]) * scale_y
                    x2 = float(boxes[idx, 2]) * scale_x
                    y2 = float(boxes[idx, 3]) * scale_y
                    raw_detections.append(
                        Detection(bbox=(x1, y1, x2, y2), score=score_f, class_id=cls_id, label=class_name)
                    )
        else:
            # Raw-map fallback: decode stride-4 heatmap, offset, radius maps in NumPy
            hm_t = _find_output_tensor(self.output_names, outputs, ["heatmap", "hm"])
            off_t = _find_output_tensor(self.output_names, outputs, ["offset", "off", "reg"])
            rad_t = _find_output_tensor(self.output_names, outputs, ["radius", "rad"])
            if hm_t is None or off_t is None or rad_t is None:
                raise ValueError(
                    f"CenterNet ONNX outputs {self.output_names} match neither decoded contract nor raw head maps."
                )
            raw_detections = self._decode_raw_maps(
                heatmap=_safe_unbatch(hm_t),
                offset=_safe_unbatch(off_t),
                radius=_safe_unbatch(rad_t),
                scale_x=scale_x,
                scale_y=scale_y,
                scale_r=scale_r,
                conf_threshold=conf_threshold,
                class_name=class_name,
            )

        return self._filter_and_suppress(
            raw_detections,
            conf_threshold=conf_threshold,
            min_radius=min_radius,
            max_radius=max_radius,
            min_box_size=min_box_size,
            max_box_size=max_box_size,
            max_aspect_ratio=max_aspect_ratio,
            target_class_id=target_class_id,
            intra_tile_nms_threshold=intra_tile_nms_threshold,
            intra_tile_metric=intra_tile_metric,
        )

    def _decode_raw_maps(
        self,
        *,
        heatmap: np.ndarray,
        offset: np.ndarray,
        radius: np.ndarray,
        scale_x: float,
        scale_y: float,
        scale_r: float,
        conf_threshold: float,
        class_name: str,
    ) -> list[Detection]:
        if heatmap.ndim == 2:
            heatmap = heatmap[None, :, :]
        num_classes, fh, fw = heatmap.shape
        stride = float(self.input_size) / float(fh)

        if float(np.min(heatmap)) < 0.0 or float(np.max(heatmap)) > 1.0:
            probs = _sigmoid(heatmap)
        else:
            probs = heatmap

        candidates: list[Detection] = []
        rad_2d = radius[0] if radius.ndim == 3 else radius
        for cls_id in range(num_classes):
            cls_hm = probs[cls_id]
            pooled = _max_pool_3x3(cls_hm)
            is_peak = (cls_hm == pooled) & (cls_hm >= conf_threshold)
            ys, xs = np.nonzero(is_peak)
            if ys.size == 0:
                continue
            peak_scores = cls_hm[ys, xs]
            order = np.argsort(-peak_scores)[: self.top_k]
            for idx in order:
                gy = int(ys[idx])
                gx = int(xs[idx])
                score_f = float(peak_scores[idx])
                dx = float(offset[0, gy, gx])
                dy = float(offset[1, gy, gx])
                r_px = max(0.0, float(rad_2d[gy, gx]) * scale_r)
                cx = (float(gx) + dx) * stride * scale_x
                cy = (float(gy) + dy) * stride * scale_y
                candidates.append(
                    Detection(
                        score=score_f,
                        class_id=cls_id,
                        label=class_name,
                        center_xy=(cx, cy),
                        radius_px=r_px,
                    )
                )
        candidates.sort(key=lambda d: -d.score)
        return candidates[: self.top_k]


class StarDistONNXModel(_BaseSpatialColonyONNXModel):
    """ONNX runner for 2D StarDist (``stardist``) star-convex polygon models."""

    model_type: str = "stardist"

    def predict(
        self,
        image: np.ndarray | Image.Image,
        *,
        conf_threshold: float = 0.35,
        min_radius: float = 1.5,
        max_radius: float = 250.0,
        min_box_size: float = 3.0,
        max_box_size: float = 500.0,
        max_aspect_ratio: float = 4.0,
        target_class_id: int | None = 0,
        class_name: str = "colony",
        intra_tile_nms_threshold: float = 0.45,
        intra_tile_metric: str = "ios",
    ) -> list[Detection]:
        """Run 2D StarDist inference on a single image or tile crop.

        Supports both baked-in decoded ONNX outputs (``boxes``, ``scores``, ``labels``,
        ``centers``, ``rays``, ``polygons``) and raw head maps (``prob``, ``rays``).
        """
        nchw, (orig_h, orig_w) = self.preprocess(image)
        outputs = self._run_session(nchw)
        scale_x = float(orig_w) / float(self.input_size)
        scale_y = float(orig_h) / float(self.input_size)
        scale_r = 0.5 * (scale_x + scale_y)

        scores_t = _find_output_tensor(self.output_names, outputs, ["scores", "pred_scores", "confs"])
        centers_t = _find_output_tensor(self.output_names, outputs, ["centers", "pred_centers"])
        rays_t = _find_output_tensor(self.output_names, outputs, ["rays", "pred_rays", "ray_dists"])
        polygons_t = _find_output_tensor(self.output_names, outputs, ["polygons", "pred_polygons"])
        boxes_t = _find_output_tensor(self.output_names, outputs, ["boxes", "pred_boxes", "bboxes"])
        labels_t = _find_output_tensor(self.output_names, outputs, ["labels", "pred_labels", "classes"])

        raw_detections: list[Detection] = []
        if scores_t is not None and centers_t is not None and (rays_t is not None or polygons_t is not None):
            scores = _safe_unbatch(scores_t).reshape(-1)
            centers = _safe_unbatch(centers_t).reshape(-1, 2)
            labels = (
                _safe_unbatch(labels_t).reshape(-1).astype(np.int64)
                if labels_t is not None
                else np.zeros_like(scores, dtype=np.int64)
            )
            rays = _safe_unbatch(rays_t) if rays_t is not None else None
            polygons = _safe_unbatch(polygons_t) if polygons_t is not None else None
            boxes = _safe_unbatch(boxes_t).reshape(-1, 4) if boxes_t is not None else None

            for idx, score_val in enumerate(scores):
                score_f = float(score_val)
                if score_f < conf_threshold:
                    continue
                cls_id = int(labels[idx])
                cx = float(centers[idx, 0]) * scale_x
                cy = float(centers[idx, 1]) * scale_y

                ray_list: list[float] | None = None
                if rays is not None:
                    ray_list = [max(0.0, float(r) * scale_r) for r in rays[idx].reshape(-1)]

                poly_list: list[tuple[float, float]] | None = None
                if polygons is not None:
                    poly_arr = polygons[idx].reshape(-1, 2)
                    poly_list = [(float(px) * scale_x, float(py) * scale_y) for px, py in poly_arr]
                elif ray_list is not None:
                    poly_list = reconstruct_star_polygon(cx, cy, ray_list)

                bbox: tuple[float, float, float, float] | None = None
                if boxes is not None and (abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6):
                    bbox = (
                        float(boxes[idx, 0]),
                        float(boxes[idx, 1]),
                        float(boxes[idx, 2]),
                        float(boxes[idx, 3]),
                    )

                raw_detections.append(
                    Detection(
                        bbox=bbox,
                        score=score_f,
                        class_id=cls_id,
                        label=class_name,
                        center_xy=(cx, cy),
                        rays=ray_list,
                        polygon=poly_list,
                    )
                )
        else:
            # Raw-map fallback: decode stride-4 probability map and ray distance field
            prob_t = _find_output_tensor(self.output_names, outputs, ["prob", "probs", "object_prob"])
            raw_rays_t = _find_output_tensor(self.output_names, outputs, ["rays", "ray_dists", "dist"])
            if prob_t is None or raw_rays_t is None:
                raise ValueError(
                    f"StarDist ONNX outputs {self.output_names} match neither decoded contract nor raw head maps."
                )
            raw_detections = self._decode_raw_maps(
                prob_map=_safe_unbatch(prob_t),
                ray_map=_safe_unbatch(raw_rays_t),
                scale_x=scale_x,
                scale_y=scale_y,
                scale_r=scale_r,
                conf_threshold=conf_threshold,
                class_name=class_name,
            )

        return self._filter_and_suppress(
            raw_detections,
            conf_threshold=conf_threshold,
            min_radius=min_radius,
            max_radius=max_radius,
            min_box_size=min_box_size,
            max_box_size=max_box_size,
            max_aspect_ratio=max_aspect_ratio,
            target_class_id=target_class_id,
            intra_tile_nms_threshold=intra_tile_nms_threshold,
            intra_tile_metric=intra_tile_metric,
        )

    def _decode_raw_maps(
        self,
        *,
        prob_map: np.ndarray,
        ray_map: np.ndarray,
        scale_x: float,
        scale_y: float,
        scale_r: float,
        conf_threshold: float,
        class_name: str,
    ) -> list[Detection]:
        prob_2d = prob_map[0] if prob_map.ndim == 3 else prob_map
        fh, _fw = prob_2d.shape
        stride = float(self.input_size) / float(fh)

        if float(np.min(prob_2d)) < 0.0 or float(np.max(prob_2d)) > 1.0:
            probs = _sigmoid(prob_2d)
        else:
            probs = prob_2d

        pooled = _max_pool_3x3(probs)
        ys, xs = np.nonzero((probs == pooled) & (probs >= conf_threshold))
        if ys.size == 0:
            return []

        peak_scores = probs[ys, xs]
        order = np.argsort(-peak_scores)[: self.top_k]
        detections: list[Detection] = []
        for idx in order:
            gy = int(ys[idx])
            gx = int(xs[idx])
            score_f = float(peak_scores[idx])
            cx = (float(gx) + 0.5) * stride * scale_x
            cy = (float(gy) + 0.5) * stride * scale_y
            ray_vals = [max(0.0, float(v) * scale_r) for v in ray_map[:, gy, gx]]
            detections.append(
                Detection(
                    score=score_f,
                    class_id=0,
                    label=class_name,
                    center_xy=(cx, cy),
                    rays=ray_vals,
                )
            )
        return detections


class CountabilityONNXModel(BaseONNXModel):
    """ONNX runner for 3-class Plate & Section Countability Classifier models."""

    model_type: str = "countability"

    def predict(
        self,
        image: np.ndarray | Image.Image,
        *,
        uncountable_threshold: float = 0.50,
        region_bbox: tuple[int, int, int, int] | None = None,
    ) -> CountabilityResult:
        """Classify a full plate or plate section into ``empty``, ``countable``, or ``uncountable``.

        Args:
            image: Input RGB plate image or section crop.
            uncountable_threshold: Threshold on ``P(uncountable)``. When
                ``P(uncountable) < uncountable_threshold``, ``is_countable`` is ``True``.
            region_bbox: Optional ``(xmin, ymin, xmax, ymax)`` pixel region for tracking section location.

        Returns:
            Populated :class:`CountabilityResult`.
        """
        nchw, _ = self.preprocess(image)
        outputs = self._run_session(nchw)

        logits_t = _find_output_tensor(self.output_names, outputs, ["logits"])
        probs_t = _find_output_tensor(self.output_names, outputs, ["probs", "probabilities", "scores"])

        if probs_t is None and logits_t is None:
            raise ValueError(
                f"Countability ONNX outputs {self.output_names} missing expected 'probs' or 'logits' tensor."
            )

        if logits_t is not None:
            logits_1d = _safe_unbatch(logits_t).reshape(-1).astype(np.float32)
        else:
            assert probs_t is not None
            probs_tmp = _safe_unbatch(probs_t).reshape(-1).astype(np.float32)
            logits_1d = np.log(np.clip(probs_tmp, 1e-7, 1.0)).astype(np.float32)

        if probs_t is not None:
            probs_1d = _safe_unbatch(probs_t).reshape(-1).astype(np.float32)
        else:
            probs_1d = _softmax_1d(logits_1d)

        if probs_1d.shape[0] < 3 or logits_1d.shape[0] < 3:
            raise ValueError(
                f"Expected 3-class countability output (empty, countable, uncountable), got shape {probs_1d.shape}."
            )

        probs_tuple = (float(probs_1d[0]), float(probs_1d[1]), float(probs_1d[2]))
        logits_tuple = (float(logits_1d[0]), float(logits_1d[1]), float(logits_1d[2]))
        pred_cls = int(np.argmax(probs_1d[:3]))
        pred_label = COUNTABILITY_LABELS[pred_cls]
        countable_prob = probs_tuple[0] + probs_tuple[1]
        uncountable_prob = probs_tuple[2]
        is_countable = uncountable_prob < float(uncountable_threshold)

        return CountabilityResult(
            probs=probs_tuple,
            logits=logits_tuple,
            pred_class_id=pred_cls,
            pred_label=pred_label,
            is_countable=is_countable,
            countable_probability=countable_prob,
            uncountable_probability=uncountable_prob,
            region_bbox=region_bbox,
        )


AnyONNXModel = CenterNetONNXModel | StarDistONNXModel | CountabilityONNXModel


def load_onnx_model(
    model_path: str | Path,
    model_type: str = "auto",
    device: str = "auto",
    input_size: int = 640,
    top_k: int = 300,
) -> AnyONNXModel:
    """Load an AutoCFU ONNX model and instantiate the matching runner class.

    Args:
        model_path: Filesystem path to an exported ``.onnx`` model.
        model_type: One of ``"auto"``, ``"centernet"``, ``"stardist"``, or ``"countability"``.
        device: Execution provider preference (``"auto"``, ``"cpu"``, or ``"cuda"``).
        input_size: Model spatial input size in pixels.
        top_k: Maximum peak candidates decoded per tile when using raw-map fallback.

    Returns:
        Instantiated :class:`CenterNetONNXModel`, :class:`StarDistONNXModel`, or
        :class:`CountabilityONNXModel`.
    """
    resolved_path = Path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {resolved_path}")

    providers = _resolve_providers(device)
    session = ort.InferenceSession(str(resolved_path), providers=providers)
    output_names = [out.name for out in session.get_outputs()]
    meta = session.get_modelmeta()
    metadata = dict(meta.custom_metadata_map) if meta and meta.custom_metadata_map else {}

    req_type = model_type.strip().lower()
    if req_type == "auto":
        req_type = detect_onnx_model_type(output_names, metadata)

    if req_type == "centernet":
        return CenterNetONNXModel(
            model_path=resolved_path,
            device=device,
            input_size=input_size,
            top_k=top_k,
            session=session,
        )
    if req_type == "stardist":
        return StarDistONNXModel(
            model_path=resolved_path,
            device=device,
            input_size=input_size,
            top_k=top_k,
            session=session,
        )
    if req_type == "countability":
        return CountabilityONNXModel(
            model_path=resolved_path,
            device=device,
            input_size=input_size,
            session=session,
        )
    raise ValueError(
        f"Unsupported model_type '{model_type}'. Expected 'auto', 'centernet', 'stardist', or 'countability'."
    )


load_model = load_onnx_model
