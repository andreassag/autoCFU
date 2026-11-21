"""Unit tests for AutoCFUEngine, mask export, overlays, and CSV/JSON reporters."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from autocfu.config import EngineConfig, InferenceConfig, SlicingConfig
from autocfu.merger import Detection
from autocfu.model import AnyONNXModel, CountabilityResult
from autocfu.pipeline import (
    AutoCFUEngine,
    InferenceResult,
    PlateCountabilityReport,
    export_batch_summary_csv,
)


class _StubSpatialModel:
    def __init__(self, model_type: str = "stardist") -> None:
        self.model_type = model_type

    def predict(self, image: Any, **kwargs: Any) -> list[Detection]:
        _ = (image, kwargs)
        if self.model_type == "stardist":
            return [Detection(score=0.92, center_xy=(320.0, 320.0), rays=[14.0] * 32)]
        return [Detection(score=0.90, center_xy=(320.0, 320.0), radius_px=12.0)]


class _StubCountabilityModel:
    model_type: str = "countability"

    def predict(
        self,
        image: Any,
        *,
        uncountable_threshold: float = 0.50,
        region_bbox: tuple[int, int, int, int] | None = None,
    ) -> CountabilityResult:
        _ = image
        is_full = region_bbox is not None and (region_bbox[2] - region_bbox[0]) > 640
        probs = (0.05, 0.85, 0.10) if is_full else (0.10, 0.30, 0.60)
        cls_id = 1 if is_full else 2
        labels = ("empty", "countable", "uncountable")
        return CountabilityResult(
            probs=probs,
            logits=(0.0, 0.0, 0.0),
            pred_class_id=cls_id,
            pred_label=labels[cls_id],
            is_countable=probs[2] < uncountable_threshold,
            countable_probability=probs[0] + probs[1],
            uncountable_probability=probs[2],
            region_bbox=region_bbox,
        )


def test_stardist_engine_mask_and_exports(tmp_path: Path) -> None:
    cfg = EngineConfig(
        slicing=SlicingConfig(slice_width=640, slice_height=640, overlap_width_ratio=0.20, overlap_height_ratio=0.20),
        inference=InferenceConfig(model_type="stardist", match_metric="center_dist", postprocess_match_threshold=0.8),
    )
    engine = AutoCFUEngine(
        model_path=tmp_path / "sd.onnx", config=cfg, model=cast(AnyONNXModel, _StubSpatialModel("stardist"))
    )

    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result = engine.predict(img)
    assert result.model_type == "stardist"
    assert result.num_slices == 4
    # Center-placed detections at tile (320, 320) become (320,320), (680,320), (320,680), (680,680)
    assert result.colony_count == 4
    assert len(result.radii) == 4
    assert all(poly is not None and len(poly) == 32 for poly in result.polygons)

    mask = result.to_mask(dtype=np.uint16)
    assert mask.shape == (1000, 1000)
    assert mask.dtype == np.uint16
    assert mask[320, 320] == 1
    assert mask[680, 680] == 4

    tif_path = result.save_mask(tmp_path / "out_mask.tif")
    png_path = result.save_mask(tmp_path / "out_mask.png")
    assert tif_path.exists()
    assert png_path.exists()
    with Image.open(tif_path) as m_img:
        loaded = np.asarray(m_img)
        assert loaded.shape == (1000, 1000)
        assert int(loaded[320, 320]) == 1

    overlay_path = tmp_path / "overlay.png"
    result.draw_overlay(img, output_path=overlay_path, show_labels=True)
    assert overlay_path.exists()

    json_path = result.save_json(tmp_path / "pred.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["colony_count"] == 4
    assert len(payload["detections"]) == 4

    csv_path = result.save_csv(tmp_path / "pred.csv")
    rows = list(csv.reader(csv_path.open("r", encoding="utf-8")))
    assert len(rows) == 5
    assert rows[0][:4] == ["instance_id", "label", "class_id", "score"]


def test_countability_engine_both_scope(tmp_path: Path) -> None:
    cfg = EngineConfig(
        slicing=SlicingConfig(slice_width=640, slice_height=640, overlap_width_ratio=0.20, overlap_height_ratio=0.20),
        inference=InferenceConfig(model_type="countability", countability_scope="both"),
    )
    engine = AutoCFUEngine(
        model_path=tmp_path / "cnt.onnx", config=cfg, model=cast(AnyONNXModel, _StubCountabilityModel())
    )

    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result = engine.predict(img)
    assert result.model_type == "countability"
    assert result.is_countable is True
    assert result.countability_report is not None
    assert result.countability_report.total_sections == 4
    assert result.countability_report.uncountable_section_count == 4

    overlay_path = tmp_path / "cnt_overlay.png"
    result.draw_overlay(img, output_path=overlay_path)
    assert overlay_path.exists()

    json_path = result.save_json(tmp_path / "cnt.json")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["countability"]["is_countable"] is True

    csv_path = result.save_csv(tmp_path / "cnt.csv")
    rows = list(csv.reader(csv_path.open("r", encoding="utf-8")))
    assert len(rows) == 6

    summary_csv = export_batch_summary_csv([result], tmp_path / "summary.csv")
    sum_rows = list(csv.reader(summary_csv.open("r", encoding="utf-8")))
    assert len(sum_rows) == 2


def test_plate_countability_report_sections_only() -> None:
    sec_cnt = CountabilityResult((0.1, 0.8, 0.1), (0.0, 0.0, 0.0), 1, "countable", True, 0.9, 0.1, (0, 0, 640, 640))
    sec_unc = CountabilityResult(
        (0.0, 0.1, 0.9), (0.0, 0.0, 0.0), 2, "uncountable", False, 0.1, 0.9, (640, 0, 1280, 640)
    )
    rep = PlateCountabilityReport(whole_plate=None, section_predictions=[sec_cnt, sec_unc], uncountable_threshold=0.4)
    # Mean P(uncountable) = 0.50 >= 0.40 => False
    assert rep.is_countable is False
    res = InferenceResult(image_path=None, image_size=(1280, 640), model_type="countability", countability_report=rep)
    assert res.is_countable is False
