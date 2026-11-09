"""Unit tests for calculate_iou, GreedyNMM, and NMS mergers."""

from __future__ import annotations

import tests  # noqa: F401  # isort: skip

import unittest

from autocfu.merger import NMS, Detection, GreedyNMM, calculate_ios, calculate_iou, get_merger


class TestMerger(unittest.TestCase):
    def test_calculate_iou(self) -> None:
        # Identical boxes
        box_a = [0.0, 0.0, 10.0, 10.0]
        self.assertAlmostEqual(calculate_iou(box_a, box_a), 1.0)

        # Disjoint boxes
        box_b = [20.0, 20.0, 30.0, 30.0]
        self.assertAlmostEqual(calculate_iou(box_a, box_b), 0.0)

        # Partial overlap: Box A [0, 0, 10, 10] (area 100), Box C [5, 0, 15, 10] (area 100)
        # Intersection: [5, 0, 10, 10] -> width 5, height 10 -> area 50
        # Union: 100 + 100 - 50 = 150 -> IoU = 50 / 150 = 1/3
        box_c = [5.0, 0.0, 15.0, 10.0]
        self.assertAlmostEqual(calculate_iou(box_a, box_c), 1.0 / 3.0)

        # Zero-area box
        empty_box = [5.0, 5.0, 5.0, 5.0]
        self.assertAlmostEqual(calculate_iou(box_a, empty_box), 0.0)

    def test_greedy_nmm_merging(self) -> None:
        """Test that overlapping boxes are merged with score-weighted centroid coordinates."""
        merger = GreedyNMM(iou_threshold=0.50)

        # Box 1: [10, 10, 30, 30] score 0.8
        # Box 2: [12, 12, 32, 32] score 0.2
        # Intersection: [12, 12, 30, 30] -> width 18, height 18 -> 324
        # Area 1: 400, Area 2: 400 -> Union: 800 - 324 = 476 -> IoU = 324 / 476 = 0.6806 > 0.50
        det1 = Detection(bbox=(10.0, 10.0, 30.0, 30.0), score=0.8, class_id=0, label="colony")
        det2 = Detection(bbox=(12.0, 12.0, 32.0, 32.0), score=0.2, class_id=0, label="colony")

        merged = merger.merge([det1, det2])
        self.assertEqual(len(merged), 1)

        result = merged[0]
        self.assertEqual(result.class_id, 0)
        self.assertAlmostEqual(result.score, 0.8)

        # Expected weighted coordinates:
        # x1 = (10.0 * 0.8 + 12.0 * 0.2) / 1.0 = 8.0 + 2.4 = 10.4
        # y1 = (10.0 * 0.8 + 12.0 * 0.2) / 1.0 = 10.4
        # x2 = (30.0 * 0.8 + 32.0 * 0.2) / 1.0 = 24.0 + 6.4 = 30.4
        # y2 = (30.0 * 0.8 + 32.0 * 0.2) / 1.0 = 30.4
        self.assertAlmostEqual(result.x1, 10.4)
        self.assertAlmostEqual(result.y1, 10.4)
        self.assertAlmostEqual(result.x2, 30.4)
        self.assertAlmostEqual(result.y2, 30.4)

    def test_greedy_nmm_disjoint_boxes(self) -> None:
        """Detections far apart should not be merged."""
        merger = GreedyNMM(iou_threshold=0.50)
        det1 = Detection(bbox=(10.0, 10.0, 30.0, 30.0), score=0.9, class_id=0, label="colony")
        det2 = Detection(bbox=(100.0, 100.0, 120.0, 120.0), score=0.85, class_id=0, label="colony")

        merged = merger.merge([det1, det2])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].score, 0.9)
        self.assertEqual(merged[1].score, 0.85)

    def test_greedy_nmm_different_classes(self) -> None:
        """Overlapping detections of different classes should remain separate by default."""
        merger = GreedyNMM(iou_threshold=0.30, class_agnostic=False)
        det1 = Detection(bbox=(10.0, 10.0, 30.0, 30.0), score=0.9, class_id=0, label="colony")
        det2 = Detection(bbox=(10.0, 10.0, 30.0, 30.0), score=0.8, class_id=1, label="defect")

        merged = merger.merge([det1, det2])
        self.assertEqual(len(merged), 2)

    def test_standard_nms(self) -> None:
        """Standard NMS should suppress the lower score candidate without altering coordinates."""
        nms = NMS(iou_threshold=0.50)
        det1 = Detection(bbox=(10.0, 10.0, 30.0, 30.0), score=0.95, class_id=0, label="colony")
        det2 = Detection(bbox=(11.0, 11.0, 31.0, 31.0), score=0.70, class_id=0, label="colony")

        survivors = nms.merge([det1, det2])
        self.assertEqual(len(survivors), 1)
        self.assertEqual(survivors[0].bbox, (10.0, 10.0, 30.0, 30.0))
        self.assertEqual(survivors[0].score, 0.95)

    def test_empty_and_single_input(self) -> None:
        merger = GreedyNMM()
        self.assertEqual(merger.merge([]), [])

        single = [Detection(bbox=(0.0, 0.0, 10.0, 10.0), score=0.9, class_id=0, label="colony")]
        merged = merger.merge(single)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].bbox, (0.0, 0.0, 10.0, 10.0))

    def test_get_merger_factory(self) -> None:
        m1 = get_merger("greedy_nmm", 0.45)
        self.assertIsInstance(m1, GreedyNMM)
        self.assertEqual(m1.iou_threshold, 0.45)

        m2 = get_merger("nms", 0.60)
        self.assertIsInstance(m2, NMS)
        self.assertEqual(m2.iou_threshold, 0.60)

    def test_match_threshold_alias(self) -> None:
        """Verify match_threshold keyword parameter acts as an alias for iou_threshold."""
        gnmm = GreedyNMM(match_threshold=0.55)
        self.assertEqual(gnmm.iou_threshold, 0.55)

        nms = NMS(match_threshold=0.65)
        self.assertEqual(nms.iou_threshold, 0.65)

        factory_gnmm = get_merger("greedy_nmm", match_threshold=0.48)
        self.assertEqual(factory_gnmm.iou_threshold, 0.48)

    def test_calculate_ios(self) -> None:
        # Box A: [0, 0, 10, 10] (area 100)
        # Box B: [0, 0, 5, 5] (area 25, fully inside Box A)
        box_a = [0.0, 0.0, 10.0, 10.0]
        box_b = [0.0, 0.0, 5.0, 5.0]
        # IoS = intersection / min_area = 25 / 25 = 1.0
        self.assertAlmostEqual(calculate_ios(box_a, box_b), 1.0)
        # Disjoint
        box_c = [20.0, 20.0, 30.0, 30.0]
        self.assertAlmostEqual(calculate_ios(box_a, box_c), 0.0)

    def test_ios_merging(self) -> None:
        # One large box and one small box completely inside it
        # IoU would be 25 / 100 = 0.25 (below 0.5)
        # IoS is 25 / 25 = 1.0 (above 0.5)
        det_large = Detection(bbox=(0.0, 0.0, 10.0, 10.0), score=0.9, class_id=0, label="colony")
        det_small = Detection(bbox=(0.0, 0.0, 5.0, 5.0), score=0.8, class_id=0, label="colony")

        merger_iou = GreedyNMM(iou_threshold=0.50, match_metric="iou")
        self.assertEqual(len(merger_iou.merge([det_large, det_small])), 2)

        merger_ios = GreedyNMM(iou_threshold=0.50, match_metric="ios")
        self.assertEqual(len(merger_ios.merge([det_large, det_small])), 1)

    def test_invalid_match_metric(self) -> None:
        with self.assertRaises(ValueError):
            GreedyNMM(match_metric="invalid")
        with self.assertRaises(ValueError):
            NMS(match_metric="invalid")

    def test_detection_spatial_properties(self) -> None:
        det = Detection(bbox=(10.0, 20.0, 50.0, 60.0), score=0.95, class_id=0, label="colony")
        self.assertAlmostEqual(det.width, 40.0)
        self.assertAlmostEqual(det.height, 40.0)
        self.assertAlmostEqual(det.area, 1600.0)
        self.assertAlmostEqual(det.cx, 30.0)
        self.assertAlmostEqual(det.cy, 40.0)
        self.assertEqual(det.center, (30.0, 40.0))
        self.assertAlmostEqual(det.radius, 20.0)

        d = det.to_dict()
        self.assertEqual(d["center"], [30.0, 40.0])
        self.assertEqual(d["bbox"], [10.0, 20.0, 50.0, 60.0])


if __name__ == "__main__":
    unittest.main()
