"""Unit tests for AutoCFUEngine pipeline, config, and output serializers."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from autocfu.config import EngineConfig, load_config
from autocfu.merger import Detection
from autocfu.pipeline import AgarInferenceEngine, AutoCFUEngine, InferenceResult, export_batch_summary_csv


class MockModel:
    """Mock model that returns deterministic detections for pipeline testing."""

    def __init__(self, detections_per_slice: List[Detection]) -> None:
        self.detections = detections_per_slice

    def predict_slice(
        self,
        crop: np.ndarray,
        confidence_threshold: float = 0.35,
        target_class_id: int = 0,
        class_name: str = "colony",
    ) -> List[Detection]:
        # Return detections that meet confidence threshold and target class
        return [
            d
            for d in self.detections
            if d.score >= confidence_threshold and (target_class_id is None or d.class_id == target_class_id)
        ]


class TestPipeline(unittest.TestCase):
    def test_config_load_and_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            cfg = EngineConfig()
            cfg.slicing.slice_width = 500
            cfg.inference.confidence_threshold = 0.42
            cfg.save_yaml(yaml_path)

            loaded = load_config(yaml_path)
            self.assertEqual(loaded.slicing.slice_width, 500)
            self.assertAlmostEqual(loaded.inference.confidence_threshold, 0.42)

    def test_pipeline_with_mock_model(self) -> None:
        # Define mock detection inside slice coordinates (e.g., at [10, 10, 30, 30])
        mock_detection = Detection(
            bbox=(10.0, 10.0, 30.0, 30.0),
            score=0.9,
            class_id=0,
            label="colony",
        )
        mock_model = MockModel([mock_detection])

        cfg = EngineConfig()
        cfg.slicing.slice_width = 200
        cfg.slicing.slice_height = 200
        cfg.slicing.overlap_width_ratio = 0.0
        cfg.slicing.overlap_height_ratio = 0.0

        engine = AutoCFUEngine(model=mock_model, config=cfg)
        self.assertIs(AgarInferenceEngine, AutoCFUEngine)

        # Create a synthetic 400x400 image
        synthetic_img = Image.new("RGB", (400, 400), color=(200, 200, 200))
        result = engine.infer_image(synthetic_img)

        self.assertEqual(result.num_slices, 4)
        # Because the mock returns 1 detection per slice, and slices are disjoint:
        self.assertEqual(result.colony_count, 4)
        self.assertEqual(result.image_size, (400, 400))
        self.assertGreater(result.execution_time_seconds, 0.0)

        # Verify coordinate shifting on the 4 tiles:
        # Tiles start at (0,0), (200,0), (0,200), (200,200)
        # Detection at [10, 10, 30, 30] shifted to:
        # Tile 1: [10, 10, 30, 30]
        # Tile 2: [210, 10, 230, 30]
        # Tile 3: [10, 210, 30, 230]
        # Tile 4: [210, 210, 230, 230]
        bboxes = sorted([det.bbox for det in result.detections])
        self.assertEqual(bboxes[0], (10.0, 10.0, 30.0, 30.0))
        self.assertEqual(bboxes[1], (10.0, 210.0, 30.0, 230.0))
        self.assertEqual(bboxes[2], (210.0, 10.0, 230.0, 30.0))
        self.assertEqual(bboxes[3], (210.0, 210.0, 230.0, 230.0))

    def test_result_serialization_and_overlay(self) -> None:
        det = Detection(bbox=(50.0, 50.0, 100.0, 100.0), score=0.88, class_id=0, label="colony")
        res = InferenceResult(
            image_path="test_dish.jpg",
            image_size=(500, 500),
            detections=[det],
            colony_count=1,
            num_slices=1,
            execution_time_seconds=0.123,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "output.json"
            res.save_json(json_file)
            self.assertTrue(json_file.is_file())

            with open(json_file, "r") as f:
                data = json.load(f)
            self.assertEqual(data["colony_count"], 1)
            self.assertEqual(data["image_width"], 500)

            csv_file = Path(tmpdir) / "output.csv"
            res.save_csv(csv_file)
            self.assertTrue(csv_file.is_file())

            batch_csv = Path(tmpdir) / "batch.csv"
            export_batch_summary_csv([res], batch_csv)
            self.assertTrue(batch_csv.is_file())

            overlay_img = Path(tmpdir) / "overlay.png"
            res.draw_overlay(output_path=overlay_img)
            self.assertTrue(overlay_img.is_file())

    def test_spatial_helpers_and_filtering(self) -> None:
        det1 = Detection(bbox=(10.0, 10.0, 30.0, 30.0), score=0.9, class_id=0, label="colony")  # center (20, 20)
        det2 = Detection(bbox=(100.0, 100.0, 120.0, 120.0), score=0.8, class_id=0, label="colony")  # center (110, 110)
        res = InferenceResult(
            image_path=None,
            image_size=(200, 200),
            detections=[det1, det2],
            colony_count=2,
            num_slices=1,
            execution_time_seconds=0.01,
        )

        # Test helper properties
        self.assertEqual(res.bboxes, [(10.0, 10.0, 30.0, 30.0), (100.0, 100.0, 120.0, 120.0)])
        self.assertEqual(res.centers, [(20.0, 20.0), (110.0, 110.0)])
        self.assertEqual(res.scores, [0.9, 0.8])

        # Test numpy conversions
        arr_5 = res.to_numpy(include_score=True)
        self.assertEqual(arr_5.shape, (2, 5))
        self.assertEqual(arr_5[0, 4], 0.9)

        centers_arr = res.centers_numpy()
        self.assertEqual(centers_arr.shape, (2, 2))
        self.assertEqual(centers_arr[0, 0], 20.0)
        self.assertEqual(centers_arr[0, 1], 20.0)

        # Test ROI filtering (e.g. drop quadrant 1)
        roi_dets = res.filter_by_roi(x1=0.0, y1=0.0, x2=50.0, y2=50.0)
        self.assertEqual(len(roi_dets), 1)
        self.assertEqual(roi_dets[0].center, (20.0, 20.0))

        # Test circle filtering (e.g. pipetted spot centered at (20, 20) with radius 15)
        spot_dets = res.filter_by_circle(cx=20.0, cy=20.0, radius=15.0)
        self.assertEqual(len(spot_dets), 1)
        self.assertEqual(spot_dets[0].center, (20.0, 20.0))

        # Circle that doesn't cover either
        empty_spot = res.filter_by_circle(cx=50.0, cy=50.0, radius=5.0)
        self.assertEqual(len(empty_spot), 0)

    def test_return_raw_detections(self) -> None:
        mock_model = MockModel([Detection(bbox=(10.0, 10.0, 30.0, 30.0), score=0.9, class_id=0, label="colony")])
        cfg = EngineConfig()
        cfg.slicing.slice_width = 200
        cfg.slicing.slice_height = 200
        engine = AutoCFUEngine(model=mock_model, config=cfg)

        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        res_with_raw = engine.infer_image(img, return_raw=True)
        self.assertIsNotNone(res_with_raw.raw_detections)
        self.assertEqual(len(res_with_raw.raw_detections), 1)

        res_without_raw = engine.infer_image(img, return_raw=False)
        self.assertIsNone(res_without_raw.raw_detections)


if __name__ == "__main__":
    unittest.main()
