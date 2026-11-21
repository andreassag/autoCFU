"""Unit tests for dish metadata helpers, GT extraction, and metric calculation."""

from __future__ import annotations

import numpy as np
from PIL import Image

from autocfu.merger import Detection
from autocfu.utils import (
    compute_detection_metrics,
    extract_gt_colony_boxes,
    is_colony_label,
    is_valid_highres_plate,
    plot_plate_detections,
)


def test_is_colony_label() -> None:
    assert is_colony_label("colony")
    assert is_colony_label("E.coli")
    assert is_colony_label(0)
    assert not is_colony_label("contamination")
    assert not is_colony_label("background")
    assert not is_colony_label(None)


def test_is_valid_highres_plate() -> None:
    assert is_valid_highres_plate({"width": 4000, "height": 4000, "is_tiled": False})
    assert not is_valid_highres_plate({"width": 4000, "height": 4000, "is_tiled": True})
    assert not is_valid_highres_plate({"width": 640, "height": 640})


def test_extract_gt_colony_boxes_centernet_schema() -> None:
    sample = {
        "objects": [
            {"x_px": 100.0, "y_px": 200.0, "r_px": 15.0, "single_class_name": "colony", "is_contamination": False},
            {"x_px": 300.0, "y_px": 300.0, "r_px": 20.0, "single_class_name": "colony", "is_contamination": True},
        ]
    }
    boxes = extract_gt_colony_boxes(sample)
    assert len(boxes) == 1
    assert boxes[0] == (85.0, 185.0, 115.0, 215.0)


def test_compute_detection_metrics() -> None:
    preds = [
        Detection(bbox=(10.0, 10.0, 30.0, 30.0), score=0.95),
        Detection(bbox=(200.0, 200.0, 220.0, 220.0), score=0.70),
    ]
    gt = [(11.0, 11.0, 31.0, 31.0), (50.0, 50.0, 70.0, 70.0)]
    metrics = compute_detection_metrics(preds, gt, iou_threshold=0.5)
    assert metrics["tp"] == 1.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["count_error"] == 0.0


def test_plot_plate_detections(tmp_path) -> None:
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    dets = [
        Detection(score=0.9, center_xy=(64.0, 64.0), rays=[12.0] * 16),
        Detection(score=0.8, center_xy=(20.0, 20.0), radius_px=8.0),
    ]
    out_file = tmp_path / "plot.png"
    rendered = plot_plate_detections(
        img, dets, gt_boxes=[(50.0, 50.0, 78.0, 78.0)], output_path=out_file, show_labels=True
    )
    assert isinstance(rendered, Image.Image)
    assert out_file.exists()
