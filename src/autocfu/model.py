"""ONNX Runtime session wrapper for RT-DETRv4 colony detection models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .merger import Detection

try:
    import onnxruntime as _ort
except ImportError:
    _ort = None

ort: Any = _ort


def _safe_unbatch(arr: np.ndarray, target_min_dim: int) -> np.ndarray:
    """Safely strip leading batch dimension if batch size is 1.

    Args:
        arr: Array-like input tensor.
        target_min_dim: Minimum number of dimensions expected for unbatched tensor.

    Returns:
        NumPy array with leading batch dimension removed if redundant.
    """
    arr = np.asarray(arr)
    if arr.ndim > target_min_dim and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _find_output_tensor(
    output_dict: dict[str, Any],
    raw_outputs: Sequence[Any],
    candidates: Sequence[str],
    fallback_index: int,
) -> np.ndarray:
    """Locate an output tensor by possible name candidates or fallback index.

    Attempts to locate the target tensor first by exact candidate name matching,
    then by case-insensitive candidate name matching in `output_dict`. If no candidate
    name matches, falls back to positional indexing within `raw_outputs`.

    Args:
        output_dict: Mapping of output node names to output tensors.
        raw_outputs: Sequence of raw output tensors returned by the inference session.
        candidates: Preferred output node names in descending priority order.
        fallback_index: 0-based positional index in `raw_outputs` if candidate names match fails.

    Returns:
        The matched NumPy array or tensor output.

    Raises:
        ValueError: If `raw_outputs` is empty and no candidate tensor could be found.
    """
    for name in candidates:
        if name in output_dict and output_dict[name] is not None:
            return output_dict[name]

    lower_dict = {k.lower(): v for k, v in output_dict.items() if v is not None}
    for name in candidates:
        if name.lower() in lower_dict:
            return lower_dict[name.lower()]

    if 0 <= fallback_index < len(raw_outputs):
        return raw_outputs[fallback_index]
    if len(raw_outputs) > 0:
        return raw_outputs[0]

    raise ValueError("Cannot locate output tensor: `raw_outputs` is empty and no candidates matched.")


class RTDetrV4ONNXModel:
    """Wrapper managing ONNX Runtime inference sessions for RT-DETRv4 models."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "auto",
        input_size: int = 640,
        session: Any | None = None,
    ) -> None:
        global ort
        if ort is None and session is None:
            try:
                import onnxruntime as ort
            except ImportError:
                pass

        if ort is None and session is None:
            raise ImportError("onnxruntime is required for model inference. Install it via `pip install onnxruntime`.")

        self.input_size = input_size
        self.device = device.lower()

        if session is not None:
            self.session = session
            self.model_path = Path(model_path) if model_path else Path("custom_session.onnx")
            self.active_provider = (
                self.session.get_providers()[0] if hasattr(self.session, "get_providers") else "Custom"
            )
        elif model_path is not None:
            if ort is None:
                raise ImportError(
                    "onnxruntime is required for model inference. Install it via `pip install onnxruntime`."
                )
            self.model_path = Path(model_path)
            if not self.model_path.is_file():
                raise FileNotFoundError(f"ONNX model file not found at: {self.model_path}")

            # Determine execution providers
            providers = self._resolve_providers(self.device)
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            try:
                self.session = ort.InferenceSession(
                    str(self.model_path), sess_options=sess_options, providers=providers
                )
            except Exception as err:
                # Fallback to CPU if CUDA provider failed
                if "CUDAExecutionProvider" in providers:
                    self.session = ort.InferenceSession(
                        str(self.model_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
                    )
                else:
                    raise err

            self.active_provider = self.session.get_providers()[0]
        else:
            raise ValueError("Either 'model_path' or 'session' must be provided.")

        if hasattr(self.session, "get_inputs"):
            self._input_names = [inp.name for inp in self.session.get_inputs()]
        else:
            self._input_names = ["images"]

        if hasattr(self.session, "get_outputs"):
            self._output_names = [out.name for out in self.session.get_outputs()]
        else:
            self._output_names = []

    @staticmethod
    def _resolve_providers(device: str) -> list[str]:
        available = ort.get_available_providers() if hasattr(ort, "get_available_providers") else []
        if device == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("CUDA execution requested but CUDAExecutionProvider is unavailable.")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            return ["CPUExecutionProvider"]
        else:  # "auto"
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]

    def predict_slice(
        self,
        crop: np.ndarray | Image.Image,
        confidence_threshold: float = 0.35,
        target_class_id: int | None = 0,
        class_name: str = "colony",
        intra_tile_nms_threshold: float = 0.0,
        intra_tile_match_metric: str = "ios",
        min_box_size: float = 0.0,
        max_box_size: float = 0.0,
        max_aspect_ratio: float = 0.0,
    ) -> list[Detection]:
        """Perform object detection on an individual image tile/slice.

        Args:
            crop: Tile image as numpy RGB array or PIL Image.
            confidence_threshold: Minimum prediction score.
            target_class_id: If specified, filter detections to this class ID.
            class_name: Human-readable label name for detected objects.
            intra_tile_nms_threshold: Metric threshold for intra-tile duplicate query suppression (0.0 disables).
            intra_tile_match_metric: Overlap metric for intra-tile suppression ('iou' or 'ios').
            min_box_size: Minimum width or height in pixels (0.0 disables).
            max_box_size: Maximum width or height in pixels (0.0 disables).
            max_aspect_ratio: Maximum aspect ratio max(w/h, h/w) allowed (0.0 disables). Also suppresses
                elongated border rim-glare slivers.

        Returns:
            List of Detection objects in coordinates relative to the slice [0, crop_w/h].
        """
        # Convert input to PIL Image for standardized handling and resizing
        if isinstance(crop, Image.Image):
            pil_img = crop.convert("RGB")
        elif isinstance(crop, np.ndarray):
            pil_img = Image.fromarray(crop).convert("RGB")
        else:
            raise TypeError(f"Expected PIL Image or numpy array, got {type(crop)}")

        orig_w, orig_h = pil_img.size
        if orig_w == 0 or orig_h == 0:
            return []

        if orig_w != self.input_size or orig_h != self.input_size:
            resized_img = pil_img.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        else:
            resized_img = pil_img

        # Normalize to float32 [0.0, 1.0] and rearrange to CHW
        img_np = np.asarray(resized_img, dtype=np.float32) / 255.0
        # Transpose from (H, W, C) to (C, H, W) and add batch dimension (1, C, H, W)
        blob = np.transpose(img_np, (2, 0, 1))[None, ...]

        # Prepare model inputs
        feed_dict: dict[str, Any] = {}
        if "images" in self._input_names:
            feed_dict["images"] = blob
        elif self._input_names:
            feed_dict[self._input_names[0]] = blob
        else:
            feed_dict["images"] = blob

        has_orig_sizes = "orig_target_sizes" in self._input_names
        if has_orig_sizes:
            # RT-DETR deployment postprocessor expects [height, width]
            feed_dict["orig_target_sizes"] = np.array([[orig_h, orig_w]], dtype=np.int64)

        raw_outputs = self.session.run(None, feed_dict)
        output_dict = dict(zip([o.lower() for o in self._output_names], raw_outputs))

        labels_raw = _find_output_tensor(output_dict, raw_outputs, ["labels", "pred_labels", "classes"], 0)
        boxes_raw = _find_output_tensor(output_dict, raw_outputs, ["boxes", "pred_boxes", "bboxes"], 1)
        scores_raw = _find_output_tensor(output_dict, raw_outputs, ["scores", "pred_scores", "confs"], 2)

        labels = _safe_unbatch(labels_raw, target_min_dim=1)
        boxes = _safe_unbatch(boxes_raw, target_min_dim=2)
        scores = _safe_unbatch(scores_raw, target_min_dim=1)

        # If model did not take orig_target_sizes and image was resized, rescale boxes
        scale_x = (orig_w / self.input_size) if (not has_orig_sizes and orig_w != self.input_size) else 1.0
        scale_y = (orig_h / self.input_size) if (not has_orig_sizes and orig_h != self.input_size) else 1.0

        detections: list[Detection] = []
        for label, box, score in zip(labels, boxes, scores):
            score_val = float(score)
            label_val = int(label)

            if score_val < confidence_threshold:
                continue

            if target_class_id is not None and label_val != target_class_id:
                continue

            bx1 = max(0.0, min(float(orig_w), float(box[0]) * scale_x))
            by1 = max(0.0, min(float(orig_h), float(box[1]) * scale_y))
            bx2 = max(0.0, min(float(orig_w), float(box[2]) * scale_x))
            by2 = max(0.0, min(float(orig_h), float(box[3]) * scale_y))

            # Discard degenerate zero-area or out-of-bounds boxes
            bw = bx2 - bx1
            bh = by2 - by1
            if bw <= 0.0 or bh <= 0.0:
                continue

            if min_box_size > 0.0 and (bw < min_box_size or bh < min_box_size):
                continue

            if max_box_size > 0.0 and (bw > max_box_size or bh > max_box_size):
                continue

            if max_aspect_ratio > 0.0:
                aspect_ratio = max(bw / bh, bh / bw)
                if aspect_ratio > max_aspect_ratio:
                    continue
                border_margin = 2.0
                touches_border = (
                    bx1 <= border_margin
                    or by1 <= border_margin
                    or bx2 >= (orig_w - border_margin)
                    or by2 >= (orig_h - border_margin)
                )
                if touches_border and max(bw, bh) > 20.0 and aspect_ratio > 1.8:
                    continue

            detections.append(
                Detection(
                    bbox=(bx1, by1, bx2, by2),
                    score=score_val,
                    class_id=label_val,
                    label=class_name,
                )
            )

        if intra_tile_nms_threshold > 0.0 and len(detections) > 1:
            from .merger import NMS

            nms = NMS(iou_threshold=intra_tile_nms_threshold, match_metric=intra_tile_match_metric)
            detections = nms.merge(detections)

        return detections


# Model aliases for RT-DETR version parity
RTDetrONNXModel = RTDetrV4ONNXModel
RTDetrV2ONNXModel = RTDetrV4ONNXModel  # Backwards compatibility alias

__all__ = [
    "RTDetrV4ONNXModel",
    "RTDetrONNXModel",
    "RTDetrV2ONNXModel",
]
