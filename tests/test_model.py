"""Unit tests for CenterNet, StarDist, and Countability ONNX model runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort
import pytest

from autocfu.model import (
    CenterNetONNXModel,
    CountabilityONNXModel,
    StarDistONNXModel,
    detect_onnx_model_type,
    load_onnx_model,
)


@dataclass
class _DummyNode:
    name: str
    shape: tuple[int | str, ...]


@dataclass
class _DummyMeta:
    custom_metadata_map: dict[str, str]


class FakeOrtSession:
    """Minimal mock mimicking ``onnxruntime.InferenceSession`` for unit tests."""

    def __init__(
        self,
        *,
        input_names: list[str],
        input_shape: tuple[int, ...],
        output_names: list[str],
        outputs: list[np.ndarray],
        metadata: dict[str, str] | None = None,
    ) -> None:
        self._inputs = [_DummyNode(name=n, shape=input_shape) for n in input_names]
        self._outputs = [_DummyNode(name=n, shape=()) for n in output_names]
        self._ret = outputs
        self._meta = _DummyMeta(custom_metadata_map=metadata or {})
        self.last_feed: dict[str, Any] = {}

    def get_inputs(self) -> list[_DummyNode]:
        return self._inputs

    def get_outputs(self) -> list[_DummyNode]:
        return self._outputs

    def get_modelmeta(self) -> _DummyMeta:
        return self._meta

    def run(self, output_names: list[str], feed: dict[str, Any]) -> list[np.ndarray]:
        _ = output_names
        self.last_feed = feed
        return self._ret


def test_detect_onnx_model_type() -> None:
    assert detect_onnx_model_type(["boxes", "scores", "labels", "centers", "radii"]) == "centernet"
    assert detect_onnx_model_type(["boxes", "scores", "labels", "centers", "rays", "polygons"]) == "stardist"
    assert detect_onnx_model_type(["logits", "probs", "pred_class", "is_countable"]) == "countability"
    assert detect_onnx_model_type(["foo"], {"autocfu_model_type": "stardist"}) == "stardist"
    with pytest.raises(ValueError, match="auto-detect"):
        detect_onnx_model_type(["unknown_a", "unknown_b"])


def test_centernet_decoded_and_rescaled_inference(tmp_path: Path) -> None:
    boxes = np.array([[[90.0, 190.0, 110.0, 210.0], [5.0, 5.0, 6.0, 6.0]]], dtype=np.float32)
    scores = np.array([[0.91, 0.10]], dtype=np.float32)
    labels = np.array([[0, 0]], dtype=np.int64)
    centers = np.array([[[100.0, 200.0], [5.5, 5.5]]], dtype=np.float32)
    radii = np.array([[10.0, 0.5]], dtype=np.float32)

    fake_sess = FakeOrtSession(
        input_names=["images"],
        input_shape=(1, 3, 640, 640),
        output_names=["boxes", "scores", "labels", "centers", "radii"],
        outputs=[boxes, scores, labels, centers, radii],
    )
    model = CenterNetONNXModel(
        model_path=tmp_path / "cn.onnx", input_size=640, session=cast(ort.InferenceSession, fake_sess)
    )

    # Test with 320x320 input -> coordinates rescale by 0.5
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    dets = model.predict(img, conf_threshold=0.5, min_radius=1.5)
    assert len(dets) == 1
    assert dets[0].center_xy == pytest.approx((50.0, 100.0))
    assert dets[0].radius_px == pytest.approx(5.0)
    assert fake_sess.last_feed["images"].shape == (1, 3, 640, 640)


def test_centernet_raw_heatmap_fallback(tmp_path: Path) -> None:
    hm = np.zeros((1, 1, 160, 160), dtype=np.float32)
    off = np.zeros((1, 2, 160, 160), dtype=np.float32)
    rad = np.zeros((1, 1, 160, 160), dtype=np.float32)
    hm[0, 0, 25, 40] = 0.88
    off[0, 0, 25, 40] = 0.5
    off[0, 1, 25, 40] = 0.25
    rad[0, 0, 25, 40] = 12.0

    fake_sess = FakeOrtSession(
        input_names=["images"],
        input_shape=(1, 3, 640, 640),
        output_names=["heatmap", "offset", "radius"],
        outputs=[hm, off, rad],
    )
    model = CenterNetONNXModel(model_path=tmp_path / "cn_raw.onnx", session=cast(ort.InferenceSession, fake_sess))
    dets = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), conf_threshold=0.5)
    assert len(dets) == 1
    assert dets[0].center_xy == pytest.approx((40.5 * 4.0, 25.25 * 4.0))
    assert dets[0].radius_px == pytest.approx(12.0)


def test_stardist_decoded_and_raw_fallback(tmp_path: Path) -> None:
    num_rays = 32
    rays = np.full((1, 1, num_rays), 15.0, dtype=np.float32)
    angles = np.arange(num_rays, dtype=np.float32) * (2.0 * np.pi / float(num_rays))
    cx, cy = 200.0, 300.0
    polys = np.stack([cx + 15.0 * np.cos(angles), cy + 15.0 * np.sin(angles)], axis=-1)[None, None, :, :]
    boxes = np.array([[[cx - 15.0, cy - 15.0, cx + 15.0, cy + 15.0]]], dtype=np.float32)
    scores = np.array([[0.95]], dtype=np.float32)
    labels = np.array([[0]], dtype=np.int64)
    centers = np.array([[[cx, cy]]], dtype=np.float32)

    fake_sess = FakeOrtSession(
        input_names=["images"],
        input_shape=(1, 3, 640, 640),
        output_names=["boxes", "scores", "labels", "centers", "rays", "polygons"],
        outputs=[boxes, scores, labels, centers, rays, polys.astype(np.float32)],
    )
    model = StarDistONNXModel(model_path=tmp_path / "sd.onnx", session=cast(ort.InferenceSession, fake_sess))
    dets = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), conf_threshold=0.5)
    assert len(dets) == 1
    assert dets[0].rays is not None and len(dets[0].rays) == 32
    assert dets[0].polygon is not None and len(dets[0].polygon) == 32
    assert dets[0].radius_px == pytest.approx(15.0)

    # Also test raw StarDist maps fallback
    prob_map = np.zeros((1, 1, 160, 160), dtype=np.float32)
    ray_map = np.full((1, 16, 160, 160), 10.0, dtype=np.float32)
    prob_map[0, 0, 50, 50] = 0.85
    raw_sess = FakeOrtSession(
        input_names=["images"],
        input_shape=(1, 3, 640, 640),
        output_names=["prob", "rays"],
        outputs=[prob_map, ray_map],
    )
    raw_model = StarDistONNXModel(model_path=tmp_path / "sd_raw.onnx", session=cast(ort.InferenceSession, raw_sess))
    raw_dets = raw_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), conf_threshold=0.5)
    assert len(raw_dets) == 1
    assert raw_dets[0].polygon is not None and len(raw_dets[0].polygon) == 16


def test_countability_prediction_and_load_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    logits = np.array([[0.1, 2.0, -1.0]], dtype=np.float32)
    probs = np.array([[0.12, 0.83, 0.05]], dtype=np.float32)
    fake_sess = FakeOrtSession(
        input_names=["images"],
        input_shape=(1, 3, 640, 640),
        output_names=["logits", "probs", "pred_class", "is_countable"],
        outputs=[logits, probs, np.array([1], dtype=np.int64), np.array([True])],
    )

    dummy_model = tmp_path / "countability.onnx"
    dummy_model.write_bytes(b"mock-onnx")

    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: fake_sess)
    loaded = load_onnx_model(dummy_model, model_type="auto")
    assert isinstance(loaded, CountabilityONNXModel)
    res = loaded.predict(np.zeros((640, 640, 3), dtype=np.uint8), uncountable_threshold=0.50)
    assert res.pred_label == "countable"
    assert res.is_countable is True
    assert res.countable_probability == pytest.approx(0.95)
    assert res.uncountable_probability == pytest.approx(0.05)

    # Tighter threshold flags it as uncountable
    strict_res = loaded.predict(np.zeros((640, 640, 3), dtype=np.uint8), uncountable_threshold=0.04)
    assert strict_res.is_countable is False
