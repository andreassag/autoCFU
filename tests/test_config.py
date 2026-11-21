"""Unit tests for AutoCFU configuration schemas and YAML parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from autocfu.config import EngineConfig, InferenceConfig, SlicingConfig, load_config


def test_default_inference_yaml_loads_cleanly() -> None:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "default_inference.yaml"
    cfg = load_config(cfg_path)
    assert cfg.slicing.slice_width == 640
    assert cfg.slicing.slice_height == 640
    assert cfg.slicing.overlap_width_ratio == pytest.approx(0.20)
    assert cfg.inference.model_type == "auto"
    assert cfg.inference.match_metric == "center_dist"
    assert cfg.inference.merger_type == "greedy_nmm"
    assert cfg.inference.countability_scope == "both"
    assert cfg.inference.uncountable_threshold == pytest.approx(0.50)


def test_slicing_validation_errors() -> None:
    with pytest.raises(ValueError, match="positive"):
        SlicingConfig(slice_width=0, slice_height=640)
    with pytest.raises(ValueError, match="overlap_width_ratio"):
        SlicingConfig(overlap_width_ratio=1.0)
    with pytest.raises(ValueError, match="overlap_height_ratio"):
        SlicingConfig(overlap_height_ratio=-0.1)


def test_inference_validation_errors() -> None:
    with pytest.raises(ValueError, match="model_type"):
        InferenceConfig(model_type="rtdetr")
    with pytest.raises(ValueError, match="device"):
        InferenceConfig(device="tpu")
    with pytest.raises(ValueError, match="intra_tile_nms_threshold"):
        InferenceConfig(intra_tile_nms_threshold=2.5)
    with pytest.raises(ValueError, match="match_metric"):
        InferenceConfig(match_metric="manhattan")
    with pytest.raises(ValueError, match="merger_type"):
        InferenceConfig(merger_type="wbf")
    with pytest.raises(ValueError, match="countability_scope"):
        InferenceConfig(countability_scope="quadrants")
    with pytest.raises(ValueError, match="confidence_threshold"):
        InferenceConfig(confidence_threshold=1.5)
    with pytest.raises(ValueError, match="radius"):
        InferenceConfig(min_radius=50.0, max_radius=10.0)


def test_load_config_none_and_missing(tmp_path: Path) -> None:
    default_cfg = load_config(None)
    assert isinstance(default_cfg, EngineConfig)

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")

    bad_yaml = tmp_path / "list.yaml"
    bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(bad_yaml)
