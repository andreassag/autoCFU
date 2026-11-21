"""Cross-language mathematical parity tests between Python ``autocfu`` and Rust ``autocfu``."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from autocfu.merger import Detection, GreedyNMM, calculate_center_distance_ratio, reconstruct_star_polygon
from autocfu.model import detect_onnx_model_type
from autocfu.slicer import PlateSlicer


def test_canonical_slicer_merger_and_polygon_parity_vectors() -> None:
    # Matches `test_slicer_edge_clamping_parity` in rust/src/lib.rs
    slicer = PlateSlicer(slice_width=640, slice_height=640, overlap_width_ratio=0.20, overlap_height_ratio=0.20)
    assert slicer.step_x == 512
    assert slicer.step_y == 512
    windows = slicer.get_slice_windows(1000, 1000)
    assert [(w.xmin, w.ymin, w.xmax, w.ymax) for w in windows] == [
        (0, 0, 640, 640),
        (360, 0, 1000, 640),
        (0, 360, 640, 1000),
        (360, 360, 1000, 1000),
    ]

    # Matches `test_greedy_nmm_center_dist_and_stardist_rays` in rust/src/lib.rs
    det_a = Detection(score=0.9, center_xy=(100.0, 200.0), rays=[10.0] * 32)
    det_b = Detection(score=0.3, center_xy=(103.0, 204.0), rays=[20.0] * 32)
    assert calculate_center_distance_ratio(det_a, det_b) == pytest.approx(0.5, abs=1e-6)

    merged = GreedyNMM(match_threshold=0.6, match_metric="center_dist").merge([det_a, det_b])
    assert len(merged) == 1
    assert merged[0].center_xy is not None
    assert merged[0].center_xy[0] == pytest.approx(100.75, abs=1e-6)
    assert merged[0].center_xy[1] == pytest.approx(201.0, abs=1e-6)
    assert merged[0].radius_px == pytest.approx(12.5, abs=1e-6)
    poly = reconstruct_star_polygon(100.75, 201.0, [12.5] * 32)
    assert merged[0].polygon is not None
    for (px, py), (rx, ry) in zip(merged[0].polygon, poly):
        assert px == pytest.approx(rx, abs=1e-6)
        assert py == pytest.approx(ry, abs=1e-6)

    # Matches `test_model_auto_detection_rules` in rust/src/lib.rs
    assert detect_onnx_model_type(["boxes", "scores", "labels", "centers", "rays", "polygons"]) == "stardist"
    assert detect_onnx_model_type(["boxes", "scores", "labels", "centers", "radii"]) == "centernet"
    assert detect_onnx_model_type(["logits", "probs", "pred_class", "is_countable"]) == "countability"


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Rust toolchain not installed in this environment")
def test_rust_crate_parity_suite_passes() -> None:
    proc = subprocess.run(
        ["cargo", "test", "--lib"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Rust tests failed:\n{proc.stdout}\n{proc.stderr}"
