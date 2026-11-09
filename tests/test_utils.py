"""Unit tests for autocfu.utils functions."""

from __future__ import annotations

import tests  # noqa: F401  # isort: skip

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from autocfu.merger import Detection
from autocfu.utils import (
    IGNORED_CATEGORIES,
    KNOWN_COLONY_LABELS,
    compute_detection_metrics,
    extract_gt_colony_boxes,
    is_colony_label,
    is_valid_highres_plate,
    plot_plate_detections,
    stream_sample_plate,
)


class TestUtils(unittest.TestCase):
    def test_is_valid_highres_plate(self) -> None:
        # Valid sample with PIL Image
        img_valid = Image.new("RGB", (4000, 4000))
        sample_valid = {
            "image": img_valid,
            "colonies_number": 42,
            "image_type": 0,
        }
        self.assertTrue(is_valid_highres_plate(sample_valid))

        # Valid sample with numpy array
        arr_valid = np.zeros((3200, 3200, 3), dtype=np.uint8)
        self.assertTrue(is_valid_highres_plate({"image": arr_valid, "colonies_number": 10}))

        # Valid sample with 2D grayscale numpy array
        arr_2d_valid = np.zeros((3200, 3200), dtype=np.uint8)
        self.assertTrue(is_valid_highres_plate({"image": arr_2d_valid, "colonies_number": 10}))

        # Invalid: 1D array
        arr_1d = np.zeros(3200, dtype=np.uint8)
        self.assertFalse(is_valid_highres_plate({"image": arr_1d, "colonies_number": 10}))

        # Valid with string count and None count
        self.assertTrue(is_valid_highres_plate({"image": img_valid, "colonies_number": "10"}))
        self.assertTrue(is_valid_highres_plate({"image": img_valid, "colonies_number": None}))

        # Invalid: colonies_number == -1 (int or string)
        self.assertFalse(is_valid_highres_plate({"image": img_valid, "colonies_number": -1}))
        self.assertFalse(is_valid_highres_plate({"image": img_valid, "colonies_number": "-1"}))

        # Invalid: image_type == 2 or "uncountable"
        self.assertFalse(is_valid_highres_plate({"image": img_valid, "image_type": 2}))
        self.assertFalse(is_valid_highres_plate({"image": img_valid, "image_type": "uncountable"}))

        # Invalid: image preview 2048x2048
        img_preview = Image.new("RGB", (2048, 2048))
        self.assertFalse(is_valid_highres_plate({"image": img_preview, "colonies_number": 10}))

        # Invalid: image dimensions < 3000
        img_small = Image.new("RGB", (1500, 1500))
        self.assertFalse(is_valid_highres_plate({"image": img_small, "colonies_number": 10}))

        # Invalid: missing image or non-image object
        self.assertFalse(is_valid_highres_plate({"colonies_number": 10}))
        self.assertFalse(is_valid_highres_plate({"image": "not_an_image", "colonies_number": 10}))

    def test_is_colony_label(self) -> None:
        # Valid colonies
        self.assertTrue(is_colony_label(0))
        self.assertTrue(is_colony_label("0"))
        self.assertTrue(is_colony_label("colony"))
        self.assertTrue(is_colony_label("s. aureus"))
        self.assertTrue(is_colony_label("contamination"))

        # Ignored defect labels
        self.assertFalse(is_colony_label(5))
        self.assertFalse(is_colony_label("5"))
        self.assertFalse(is_colony_label("Defect"))
        self.assertFalse(is_colony_label(" defect "))
        self.assertFalse(is_colony_label("agar_bubble"))

        self.assertIn("colony", KNOWN_COLONY_LABELS)

    def test_extract_gt_colony_boxes(self) -> None:
        # Standard AGAR objects schema with COCO [x, y, w, h]
        sample_agar = {
            "objects": {
                "bbox": [
                    [100, 150, 40, 50],
                    [200, 250, 30, 30],
                    [300, 350, 20, 20],  # Defect
                ],
                "category": [0, "colony", "Defect"],
            }
        }
        gt = extract_gt_colony_boxes(sample_agar)
        self.assertEqual(len(gt), 2)
        self.assertEqual(gt[0], (100.0, 150.0, 140.0, 200.0))
        self.assertEqual(gt[1], (200.0, 250.0, 230.0, 280.0))

        # Flat schema fallback with [x1, y1, x2, y2]
        sample_flat = {
            "bboxes": [
                [10.0, 20.0, 30.0, 40.0],
                [50.0, 60.0, 70.0, 80.0],
            ],
            "labels": [0, 5],  # 5 is in IGNORED_CATEGORIES
        }
        gt_flat = extract_gt_colony_boxes(sample_flat)
        self.assertEqual(len(gt_flat), 1)
        self.assertEqual(gt_flat[0], (10.0, 20.0, 30.0, 40.0))

        # Ignored labels check
        self.assertIn("defect", IGNORED_CATEGORIES)
        self.assertIn(5, IGNORED_CATEGORIES)

    def test_compute_detection_metrics_empty(self) -> None:
        # Both empty
        m_both_empty = compute_detection_metrics([], [])
        self.assertEqual(m_both_empty["tp"], 0.0)
        self.assertEqual(m_both_empty["fp"], 0.0)
        self.assertEqual(m_both_empty["fn"], 0.0)
        self.assertEqual(m_both_empty["precision"], 1.0)
        self.assertEqual(m_both_empty["recall"], 1.0)
        self.assertEqual(m_both_empty["f1"], 1.0)
        self.assertEqual(m_both_empty["abs_err"], 0.0)
        self.assertEqual(m_both_empty["count_acc"], 1.0)

        # GT empty, preds non-empty
        m_gt_empty = compute_detection_metrics([[10, 10, 20, 20]], [])
        self.assertEqual(m_gt_empty["tp"], 0.0)
        self.assertEqual(m_gt_empty["fp"], 1.0)
        self.assertEqual(m_gt_empty["precision"], 0.0)
        self.assertEqual(m_gt_empty["recall"], 0.0)
        self.assertEqual(m_gt_empty["abs_err"], 1.0)
        self.assertEqual(m_gt_empty["count_acc"], 0.0)

        # Preds empty, GT non-empty
        m_pred_empty = compute_detection_metrics([], [[10, 10, 20, 20]])
        self.assertEqual(m_pred_empty["tp"], 0.0)
        self.assertEqual(m_pred_empty["fn"], 1.0)
        self.assertEqual(m_pred_empty["recall"], 0.0)
        self.assertEqual(m_pred_empty["abs_err"], 1.0)
        self.assertEqual(m_pred_empty["count_acc"], 0.0)

    def test_compute_detection_metrics_matching(self) -> None:
        # 2 predictions, 2 GT: 1 perfect match, 1 mismatch
        preds = [[10.0, 10.0, 30.0, 30.0], [100.0, 100.0, 120.0, 120.0]]
        gt = [[10.0, 10.0, 30.0, 30.0], [500.0, 500.0, 520.0, 520.0]]
        scores = [0.95, 0.80]

        metrics = compute_detection_metrics(preds, gt, scores=scores, iou_threshold=0.50)
        self.assertEqual(metrics["tp"], 1.0)
        self.assertEqual(metrics["fp"], 1.0)
        self.assertEqual(metrics["fn"], 1.0)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_stream_sample_plate_mock(self) -> None:
        mock_ds = [
            {"image": Image.new("RGB", (1000, 1000)), "colonies_number": 5},  # Invalid (< 3000)
            {"image": Image.new("RGB", (4000, 4000)), "colonies_number": 15, "image_id": "bright_0"},  # Valid 0
            {"image": Image.new("RGB", (4000, 4000)), "colonies_number": 25, "image_id": "bright_1"},  # Valid 1
        ]

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds
        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            rgb, title = stream_sample_plate("bright", index=0)
            self.assertEqual(rgb.shape, (4000, 4000, 3))
            self.assertIn("bright_0", title)

            rgb1, title1 = stream_sample_plate("bright", index=1)
            self.assertIn("bright_1", title1)

            with self.assertRaises(RuntimeError):
                stream_sample_plate("bright", index=2)

    def test_stream_sample_plate_negative_index(self) -> None:
        with self.assertRaises(ValueError):
            stream_sample_plate("bright", index=-1)

    def test_stream_sample_plate_import_error(self) -> None:
        with patch.dict("sys.modules", {"datasets": None}):
            with self.assertRaises(ImportError):
                stream_sample_plate("bright")

    def test_plot_plate_detections(self) -> None:
        img_rgb = np.zeros((1000, 1000, 3), dtype=np.uint8)
        detections = [
            Detection(bbox=(100.0, 100.0, 120.0, 120.0), score=0.92),
            (200, 200, 230, 230, 0.85),
        ]

        mock_cv2 = MagicMock()
        mock_plt = MagicMock()
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_axes = MagicMock()
        mock_axes.ravel.return_value = [mock_ax for _ in range(9)]
        mock_plt.subplots.side_effect = [
            (mock_fig, mock_ax),
            (mock_fig, mock_axes),
        ]

        with patch.dict("sys.modules", {"cv2": mock_cv2, "matplotlib": MagicMock(), "matplotlib.pyplot": mock_plt}):
            coords = plot_plate_detections(img_rgb, detections, title="Test Plate", duration=1.23, crop_size=100)
            self.assertEqual(len(coords), 2)
            self.assertEqual(coords[0], (110, 110, 20, 20, 0.92))
            self.assertEqual(coords[1], (215, 215, 30, 30, 0.85))

    def test_plot_plate_detections_import_error(self) -> None:
        img_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.dict("sys.modules", {"cv2": None}):
            with self.assertRaises(ImportError):
                plot_plate_detections(img_rgb, [], title="Test")


if __name__ == "__main__":
    unittest.main()
