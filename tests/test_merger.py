"""Unit tests for CenterNet/StarDist Detection geometry, metrics, and mergers."""

from __future__ import annotations

import math

import pytest

from autocfu.merger import (
    NMS,
    Detection,
    GreedyNMM,
    calculate_center_distance_ratio,
    calculate_ios,
    calculate_iou,
    get_merger,
    reconstruct_star_polygon,
)


def test_detection_auto_population_modes() -> None:
    cn_det = Detection(score=0.9, center_xy=(100.0, 200.0), radius_px=15.0)
    assert cn_det.bbox == (85.0, 185.0, 115.0, 215.0)
    assert cn_det.width == pytest.approx(30.0)
    assert cn_det.height == pytest.approx(30.0)

    rays = [10.0] * 32
    sd_det = Detection(score=0.85, center_xy=(50.0, 60.0), rays=rays)
    assert sd_det.polygon is not None
    assert len(sd_det.polygon) == 32
    assert sd_det.polygon[0][0] == pytest.approx(60.0, abs=1e-6)
    assert sd_det.polygon[0][1] == pytest.approx(60.0, abs=1e-6)
    assert sd_det.polygon[8][0] == pytest.approx(50.0, abs=1e-6)
    assert sd_det.polygon[8][1] == pytest.approx(70.0, abs=1e-6)
    assert sd_det.bbox is not None
    assert sd_det.bbox[0] == pytest.approx(40.0, abs=1e-6)
    assert sd_det.bbox[2] == pytest.approx(60.0, abs=1e-6)


def test_reconstruct_star_polygon_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        reconstruct_star_polygon(0.0, 0.0, [])


def test_geometry_metrics_iou_ios_center_dist() -> None:
    box_a = (0.0, 0.0, 20.0, 20.0)
    box_b = (10.0, 10.0, 30.0, 30.0)
    assert calculate_iou(box_a, box_b) == pytest.approx(100.0 / 700.0)
    assert calculate_ios(box_a, box_b) == pytest.approx(100.0 / 400.0)

    det_a = Detection(score=0.9, center_xy=(100.0, 200.0), radius_px=10.0)
    det_b = Detection(score=0.5, center_xy=(103.0, 204.0), radius_px=20.0)
    assert calculate_center_distance_ratio(det_a, det_b) == pytest.approx(0.5)

    det_zero = Detection(bbox=(0.0, 0.0, 0.0, 0.0), score=0.5, center_xy=(0.0, 0.0), radius_px=0.0)
    assert math.isinf(calculate_center_distance_ratio(det_zero, det_a))


def test_greedy_nmm_merges_stardist_rays_and_centernet_weighted() -> None:
    det_a = Detection(score=0.9, center_xy=(100.0, 200.0), rays=[10.0] * 32)
    det_b = Detection(score=0.3, center_xy=(103.0, 204.0), rays=[20.0] * 32)

    merger = GreedyNMM(match_threshold=0.6, match_metric="center_dist")
    merged = merger.merge([det_b, det_a])
    assert len(merged) == 1
    out = merged[0]
    assert out.score == pytest.approx(0.9)
    assert out.center_xy is not None
    assert out.center_xy[0] == pytest.approx(100.75, abs=1e-6)
    assert out.center_xy[1] == pytest.approx(201.0, abs=1e-6)
    assert out.radius_px == pytest.approx(12.5, abs=1e-6)
    assert out.rays is not None and len(out.rays) == 32
    assert out.rays[0] == pytest.approx(12.5, abs=1e-6)
    assert out.polygon is not None
    assert out.polygon[0][0] == pytest.approx(113.25, abs=1e-6)


def test_nms_and_get_merger_factory() -> None:
    d1 = Detection(score=0.9, center_xy=(50.0, 50.0), radius_px=10.0)
    d2 = Detection(score=0.6, center_xy=(52.0, 50.0), radius_px=10.0)
    d3 = Detection(score=0.8, center_xy=(200.0, 200.0), radius_px=10.0)

    nms = get_merger("nms", match_threshold=0.5, match_metric="center_dist")
    assert isinstance(nms, NMS)
    kept = nms.merge([d2, d1, d3])
    assert len(kept) == 2
    assert kept[0].score == pytest.approx(0.9)
    assert kept[1].score == pytest.approx(0.8)

    assert get_merger("none", 0.5, "iou") is None
    with pytest.raises(ValueError, match="merger_type"):
        get_merger("unknown")
