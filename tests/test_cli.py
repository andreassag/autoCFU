"""Unit tests for the AutoCFU command-line interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from autocfu import cli
from autocfu.merger import Detection


def test_cli_parser_accepts_new_flags(tmp_path: Path) -> None:
    parser = cli.build_parser()
    img_path = tmp_path / "plate.png"
    model_path = tmp_path / "model.onnx"
    args = parser.parse_args(
        [
            "--image",
            str(img_path),
            "--model",
            str(model_path),
            "--model-type",
            "stardist",
            "--match-metric",
            "center_dist",
            "--min-radius",
            "2.0",
            "--max-radius",
            "180.0",
            "--countability-scope",
            "sections",
            "--uncountable-threshold",
            "0.45",
            "--save-mask",
            "--save-json",
            "--save-csv",
            "--save-overlay",
        ]
    )
    assert args.model_type == "stardist"
    assert args.match_metric == "center_dist"
    assert args.min_radius == pytest.approx(2.0)
    assert args.max_radius == pytest.approx(180.0)
    assert args.countability_scope == "sections"
    assert args.uncountable_threshold == pytest.approx(0.45)
    assert args.save_mask is True


def test_cli_end_to_end_with_stubbed_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    img_path = tmp_path / "dish.png"
    Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(img_path)
    model_path = tmp_path / "centernet.onnx"
    model_path.write_bytes(b"stub")
    out_dir = tmp_path / "outputs"

    class _StubModel:
        model_type = "centernet"

        def predict(self, image, **kwargs):
            _ = (image, kwargs)
            return [Detection(score=0.95, center_xy=(128.0, 128.0), radius_px=15.0)]

    monkeypatch.setattr("autocfu.pipeline.load_onnx_model", lambda *args, **kwargs: _StubModel())

    exit_code = cli.main(
        [
            "--image",
            str(img_path),
            "--model",
            str(model_path),
            "--model-type",
            "centernet",
            "--output-dir",
            str(out_dir),
            "--save-json",
            "--save-csv",
            "--save-mask",
            "--save-overlay",
        ]
    )
    assert exit_code == 0
    assert (out_dir / "dish_predictions.json").exists()
    assert (out_dir / "dish_predictions.csv").exists()
    assert (out_dir / "dish_mask.tif").exists()
    assert (out_dir / "dish_overlay.png").exists()


def test_cli_missing_input_returns_nonzero(tmp_path: Path) -> None:
    code = cli.main(["--image", str(tmp_path / "missing.jpg"), "--model", str(tmp_path / "m.onnx")])
    assert code == 1
