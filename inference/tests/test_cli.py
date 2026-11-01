"""Unit tests for autocfu CLI argument parsing, configuration loading, and execution."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from autocfu.cli import build_parser, find_default_config, main
from autocfu.merger import Detection
from autocfu.pipeline import InferenceResult


class TestCLI(unittest.TestCase):
    def test_parser_configuration(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.prog, "autocfu")
        args = parser.parse_args(
            [
                "--image",
                "plate.jpg",
                "--model",
                "model.onnx",
                "--confidence",
                "0.45",
                "--overlap",
                "0.25",
                "--slice-size",
                "512",
                "--merger",
                "nms",
                "--match-metric",
                "ios",
                "--match-threshold",
                "0.60",
                "--device",
                "cpu",
                "--save-overlay",
                "--save-json",
                "--save-csv",
            ]
        )
        self.assertEqual(args.image, "plate.jpg")
        self.assertEqual(args.model, "model.onnx")
        self.assertAlmostEqual(args.confidence, 0.45)
        self.assertAlmostEqual(args.overlap, 0.25)
        self.assertEqual(args.slice_size, 512)
        self.assertEqual(args.merger, "nms")
        self.assertEqual(args.match_metric, "ios")
        self.assertAlmostEqual(args.match_threshold, 0.60)
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.save_overlay)
        self.assertTrue(args.save_json)
        self.assertTrue(args.save_csv)

    def test_parser_requires_input(self) -> None:
        parser = build_parser()
        # Missing both --image and --input-dir
        with self.assertRaises(SystemExit):
            parser.parse_args(["--model", "model.onnx"])

    def test_find_default_config(self) -> None:
        cfg_path = find_default_config()
        self.assertIsNotNone(cfg_path)
        self.assertTrue(Path(cfg_path).is_file())
        self.assertTrue(cfg_path.endswith("default_inference.yaml"))

    def test_main_missing_image(self) -> None:
        ret = main(["--image", "non_existent_dish.jpg", "--model", "model.onnx"])
        self.assertEqual(ret, 1)

    def test_main_missing_input_dir(self) -> None:
        ret = main(["--input-dir", "non_existent_directory", "--model", "model.onnx"])
        self.assertEqual(ret, 1)

    def test_main_missing_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "sample.jpg"
            Image.new("RGB", (100, 100)).save(img_path)

            ret = main(["--image", str(img_path), "--model", str(Path(tmpdir) / "non_existent.onnx")])
            self.assertEqual(ret, 1)

    @patch("autocfu.cli.AutoCFUEngine")
    def test_main_successful_single_image_run(self, mock_engine_cls: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test_plate.jpg"
            Image.new("RGB", (200, 200), color=(255, 255, 255)).save(img_path)

            out_dir = Path(tmpdir) / "output"

            # Setup mock engine instance
            mock_engine = MagicMock()
            mock_engine_cls.return_value = mock_engine

            det = Detection(bbox=(10.0, 10.0, 30.0, 30.0), score=0.91, class_id=0, label="colony")
            mock_result = InferenceResult(
                image_path=str(img_path),
                image_size=(200, 200),
                detections=[det],
                colony_count=1,
                num_slices=1,
                execution_time_seconds=0.05,
            )
            mock_engine.infer_image.return_value = mock_result

            args = [
                "--image",
                str(img_path),
                "--model",
                "dummy_model.onnx",
                "--output-dir",
                str(out_dir),
                "--save-overlay",
                "--save-json",
                "--save-csv",
            ]

            ret = main(args)
            self.assertEqual(ret, 0)

            # Check that output files were generated
            self.assertTrue((out_dir / "test_plate_detected.jpg").is_file())
            self.assertTrue((out_dir / "test_plate_detections.json").is_file())
            self.assertTrue((out_dir / "test_plate_detections.csv").is_file())


if __name__ == "__main__":
    unittest.main()
