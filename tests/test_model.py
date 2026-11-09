"""Unit tests for RTDetrV4ONNXModel session wrapper and provider resolution."""

from __future__ import annotations

import tests  # noqa: F401  # isort: skip

import unittest
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from autocfu.model import RTDetrONNXModel, RTDetrV2ONNXModel, RTDetrV4ONNXModel, _find_output_tensor

try:
    import onnxruntime as _ort
except ImportError:
    _ort = None

ort: Any = _ort


class MockNode:
    def __init__(self, name: str) -> None:
        self.name = name


class MockSession:
    """Mock ONNX Runtime session returning deterministic output tensors."""

    def __init__(
        self,
        input_names: Sequence[str] = ("images", "orig_target_sizes"),
        output_names: Sequence[str] = ("labels", "boxes", "scores"),
        outputs: Sequence[np.ndarray] | None = None,
    ) -> None:
        self.input_nodes = [MockNode(n) for n in input_names]
        self.output_nodes = [MockNode(n) for n in output_names]
        self.last_feed_dict: dict[str, Any] = {}
        self.outputs: Sequence[np.ndarray]

        if outputs is not None:
            self.outputs = outputs
        else:
            # Default: 1 batch, 2 detections
            labels = np.array([[0, 1]], dtype=np.int64)
            boxes = np.array([[[10.0, 15.0, 50.0, 55.0], [100.0, 100.0, 150.0, 150.0]]], dtype=np.float32)
            scores = np.array([[0.92, 0.45]], dtype=np.float32)
            self.outputs = [labels, boxes, scores]

    def get_inputs(self) -> list[MockNode]:
        return self.input_nodes

    def get_outputs(self) -> list[MockNode]:
        return self.output_nodes

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, output_names: Any, feed_dict: dict[str, Any]) -> Sequence[np.ndarray]:
        self.last_feed_dict = feed_dict
        return self.outputs


class TestRTDetrV2ONNXModel(unittest.TestCase):
    @unittest.skipIf(ort is None, "onnxruntime is not installed")
    def test_missing_model_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            RTDetrV2ONNXModel("non_existent_model_file.onnx")

    @unittest.skipIf(ort is None, "onnxruntime is not installed")
    def test_missing_args(self) -> None:
        with self.assertRaises(ValueError):
            RTDetrV2ONNXModel()

    @unittest.skipIf(ort is None, "onnxruntime is not installed")
    def test_device_resolution(self) -> None:
        assert ort is not None
        cpu_prov = RTDetrV2ONNXModel._resolve_providers("cpu")
        self.assertEqual(cpu_prov, ["CPUExecutionProvider"])

        auto_prov = RTDetrV2ONNXModel._resolve_providers("auto")
        self.assertIn("CPUExecutionProvider", auto_prov)

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            with self.assertRaises(RuntimeError):
                RTDetrV2ONNXModel._resolve_providers("cuda")
        else:
            cuda_prov = RTDetrV2ONNXModel._resolve_providers("cuda")
            self.assertEqual(cuda_prov[0], "CUDAExecutionProvider")

    def test_predict_slice_with_mock_session(self) -> None:
        mock_sess = MockSession()
        model = RTDetrV2ONNXModel(session=mock_sess, input_size=640)

        img = Image.new("RGB", (640, 640), color=(128, 128, 128))
        dets = model.predict_slice(img, confidence_threshold=0.50, target_class_id=0)

        # Only 1 detection has score >= 0.50 and class_id == 0
        self.assertEqual(len(dets), 1)
        det = dets[0]
        self.assertEqual(det.class_id, 0)
        self.assertAlmostEqual(det.score, 0.92)
        self.assertEqual(det.bbox, (10.0, 15.0, 50.0, 55.0))

    def test_orig_target_sizes_height_width_order(self) -> None:
        """Verify that orig_target_sizes passes [height, width] as required by RT-DETR."""
        mock_sess = MockSession()
        model = RTDetrV2ONNXModel(session=mock_sess, input_size=640)

        # Non-square tile: width 600, height 400
        non_square = Image.new("RGB", (600, 400))
        model.predict_slice(non_square)

        self.assertIn("orig_target_sizes", mock_sess.last_feed_dict)
        sizes = mock_sess.last_feed_dict["orig_target_sizes"]
        # Expected: [[height, width]] -> [[400, 600]]
        self.assertEqual(sizes.tolist(), [[400, 600]])

    def test_unbatched_output_tensors(self) -> None:
        """Verify safe unbatching when tensors lack leading batch dimension (e.g. [300, 4])."""
        labels = np.array([0, 0], dtype=np.int64)
        boxes = np.array([[5.0, 5.0, 20.0, 20.0], [30.0, 30.0, 50.0, 50.0]], dtype=np.float32)
        scores = np.array([0.85, 0.75], dtype=np.float32)

        mock_sess = MockSession(outputs=[labels, boxes, scores])
        model = RTDetrV2ONNXModel(session=mock_sess)

        img = Image.new("RGB", (640, 640))
        dets = model.predict_slice(img, confidence_threshold=0.70)
        self.assertEqual(len(dets), 2)
        self.assertAlmostEqual(dets[0].score, 0.85)
        self.assertAlmostEqual(dets[1].score, 0.75)

    def test_rescaling_without_orig_target_sizes(self) -> None:
        """When model lacks orig_target_sizes input, predicted boxes should scale to image size."""
        labels = np.array([[0]], dtype=np.int64)
        # Bounding box in 640x640 space: [64, 64, 320, 320]
        boxes = np.array([[[64.0, 64.0, 320.0, 320.0]]], dtype=np.float32)
        scores = np.array([[0.90]], dtype=np.float32)

        # Session without orig_target_sizes
        mock_sess = MockSession(
            input_names=["images"],
            outputs=[labels, boxes, scores],
        )
        model = RTDetrV2ONNXModel(session=mock_sess, input_size=640)

        # Image size is 320x160 (width 320, height 160)
        # Scale X = 320 / 640 = 0.5
        # Scale Y = 160 / 640 = 0.25
        img = Image.new("RGB", (320, 160))
        dets = model.predict_slice(img)
        self.assertEqual(len(dets), 1)

        bx1, by1, bx2, by2 = dets[0].bbox
        self.assertAlmostEqual(bx1, 32.0)  # 64 * 0.5
        self.assertAlmostEqual(by1, 16.0)  # 64 * 0.25
        self.assertAlmostEqual(bx2, 160.0)  # 320 * 0.5
        self.assertAlmostEqual(by2, 80.0)  # 320 * 0.25

    def test_model_aliases(self) -> None:
        """Verify that RTDetrONNXModel and RTDetrV2ONNXModel alias RTDetrV4ONNXModel."""
        self.assertIs(RTDetrONNXModel, RTDetrV4ONNXModel)
        self.assertIs(RTDetrV2ONNXModel, RTDetrV4ONNXModel)

    def test_coordinate_clamping(self) -> None:
        """Verify predicted coordinates outside slice bounds are clamped to [0, orig_w/h]."""
        labels = np.array([0], dtype=np.int64)
        # Bounding box exceeding image boundary: negative coords and coords > 640
        boxes = np.array([[[-50.0, -20.0, 700.0, 750.0]]], dtype=np.float32)
        scores = np.array([[0.95]], dtype=np.float32)

        mock_sess = MockSession(outputs=[labels, boxes, scores])
        model = RTDetrV2ONNXModel(session=mock_sess, input_size=640)

        img = Image.new("RGB", (640, 640))
        dets = model.predict_slice(img)
        self.assertEqual(len(dets), 1)
        bx1, by1, bx2, by2 = dets[0].bbox
        self.assertEqual(bx1, 0.0)
        self.assertEqual(by1, 0.0)
        self.assertEqual(bx2, 640.0)
        self.assertEqual(by2, 640.0)

    def test_degenerate_boxes_filtered(self) -> None:
        """Verify degenerate zero-area boxes (e.g. completely out-of-bounds) are discarded."""
        labels = np.array([0, 0], dtype=np.int64)
        # First box is completely outside left boundary, second is valid
        boxes = np.array([[[-50.0, -20.0, -5.0, -10.0], [50.0, 50.0, 100.0, 100.0]]], dtype=np.float32)
        scores = np.array([[0.95, 0.90]], dtype=np.float32)

        mock_sess = MockSession(outputs=[labels, boxes, scores])
        model = RTDetrV2ONNXModel(session=mock_sess, input_size=640)

        img = Image.new("RGB", (640, 640))
        dets = model.predict_slice(img)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].bbox, (50.0, 50.0, 100.0, 100.0))

    def test_intra_tile_nms_suppression(self) -> None:
        """Verify intra-tile NMS suppresses duplicate decoder query detections within the tile."""
        labels = np.array([0, 0], dtype=np.int64)
        # Two almost identical queries for the same colony:
        boxes = np.array([[[20.0, 20.0, 40.0, 40.0], [21.0, 21.0, 41.0, 41.0]]], dtype=np.float32)
        scores = np.array([[0.90, 0.85]], dtype=np.float32)

        mock_sess = MockSession(outputs=[labels, boxes, scores])
        model = RTDetrV2ONNXModel(session=mock_sess, input_size=640)

        img = Image.new("RGB", (640, 640))
        # With intra-tile NMS enabled (threshold 0.45)
        dets_suppressed = model.predict_slice(
            img,
            confidence_threshold=0.50,
            intra_tile_nms_threshold=0.45,
            intra_tile_match_metric="ios",
        )
        self.assertEqual(len(dets_suppressed), 1)
        self.assertAlmostEqual(dets_suppressed[0].score, 0.90)

        # Without intra-tile NMS (threshold 0.0 disables)
        dets_unsuppressed = model.predict_slice(
            img,
            confidence_threshold=0.50,
            intra_tile_nms_threshold=0.0,
        )
        self.assertEqual(len(dets_unsuppressed), 2)

    def test_geometric_filtering(self) -> None:
        """Verify min_box_size, max_box_size, and max_aspect_ratio filtering."""
        labels = np.array([0, 0, 0, 0], dtype=np.int64)
        boxes = np.array(
            [
                [
                    [10.0, 10.0, 14.0, 14.0],  # 4x4: too small if min_box_size=6.0
                    [20.0, 20.0, 50.0, 50.0],  # 30x30: valid
                    [100.0, 100.0, 450.0, 450.0],  # 350x350: too large if max_box_size=300.0
                    [50.0, 50.0, 60.0, 90.0],  # 10x40 (aspect ratio 4.0): invalid if max_aspect_ratio=2.5
                ]
            ],
            dtype=np.float32,
        )
        scores = np.array([[0.9, 0.9, 0.9, 0.9]], dtype=np.float32)

        mock_sess = MockSession(outputs=[labels, boxes, scores])
        model = RTDetrV4ONNXModel(session=mock_sess, input_size=640)
        img = Image.new("RGB", (640, 640))

        # Filter with min_box_size
        dets_min = model.predict_slice(img, min_box_size=6.0)
        self.assertEqual(len(dets_min), 3)

        # Filter with max_box_size
        dets_max = model.predict_slice(img, max_box_size=300.0)
        self.assertEqual(len(dets_max), 3)

        # Filter with max_aspect_ratio
        dets_aspect = model.predict_slice(img, max_aspect_ratio=2.5)
        self.assertEqual(len(dets_aspect), 3)

        # All filters combined: only the 30x30 box remains
        dets_all = model.predict_slice(img, min_box_size=6.0, max_box_size=300.0, max_aspect_ratio=2.5)
        self.assertEqual(len(dets_all), 1)
        self.assertEqual(dets_all[0].bbox, (20.0, 20.0, 50.0, 50.0))

    def test_rim_glare_border_suppression(self) -> None:
        """Verify elongated slivers along tile boundaries are rejected when max_aspect_ratio > 0."""
        labels = np.array([0, 0, 0], dtype=np.int64)
        boxes = np.array(
            [
                [
                    [0.0, 100.0, 8.0, 140.0],  # Border sliver: bx1=0, max(bw, bh)=40 > 20, aspect_ratio=5.0 > 1.8
                    [1.5, 200.0, 9.5, 240.0],  # Border sliver within 2.0px margin: bx1=1.5
                    [100.0, 100.0, 120.0, 120.0],  # Normal colony
                ]
            ],
            dtype=np.float32,
        )
        scores = np.array([[0.9, 0.9, 0.9]], dtype=np.float32)

        mock_sess = MockSession(outputs=[labels, boxes, scores])
        model = RTDetrV4ONNXModel(session=mock_sess, input_size=640)
        img = Image.new("RGB", (640, 640))

        dets = model.predict_slice(img, max_aspect_ratio=2.5)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].bbox, (100.0, 100.0, 120.0, 120.0))

    @unittest.skipIf(ort is None, "onnxruntime is not installed")
    def test_real_onnx_model_loading(self) -> None:
        """Test loading a real physical ONNX model file from disk."""
        assert ort is not None
        assert ort.__file__ is not None
        bundled_onnx = Path(ort.__file__).parent / "datasets" / "mul_1.onnx"
        if bundled_onnx.is_file():
            model = RTDetrV2ONNXModel(model_path=bundled_onnx, device="cpu")
            self.assertEqual(model.active_provider, "CPUExecutionProvider")
            self.assertIn("X", model._input_names)

    def test_find_output_tensor_with_sequence(self) -> None:
        """Verify _find_output_tensor correctly handles candidate lookup and Sequence fallback."""
        arr0 = np.array([0, 1])
        arr1 = np.array([[10, 20, 30, 40]])
        arr2 = np.array([0.9])
        raw_tuple: Sequence[np.ndarray] = (arr0, arr1, arr2)

        # Candidate match in output_dict
        d = {"labels": arr0}
        self.assertIs(_find_output_tensor(d, raw_tuple, ["labels"], 1), arr0)

        # Fallback to index when candidate not found
        d_other = {"something_else": arr0}
        self.assertIs(_find_output_tensor(d_other, raw_tuple, ["boxes", "bboxes"], 1), arr1)

        # Fallback to index 0 when fallback_index exceeds sequence length
        self.assertIs(_find_output_tensor(d_other, raw_tuple, ["scores"], 99), arr0)

    def test_predict_slice_with_tuple_sequence_outputs(self) -> None:
        """Verify predict_slice succeeds when session returns a tuple of outputs (Sequence)."""
        labels = np.array([[0]], dtype=np.int64)
        boxes = np.array([[[10.0, 10.0, 50.0, 50.0]]], dtype=np.float32)
        scores = np.array([[0.95]], dtype=np.float32)

        # MockSession initialized with tuple of outputs instead of list
        mock_sess = MockSession(outputs=(labels, boxes, scores))
        model = RTDetrV4ONNXModel(session=mock_sess, input_size=640)

        img = Image.new("RGB", (640, 640))
        dets = model.predict_slice(img)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].bbox, (10.0, 10.0, 50.0, 50.0))
        self.assertAlmostEqual(dets[0].score, 0.95)

    def test_find_output_tensor_case_insensitivity_and_none(self) -> None:
        """Verify _find_output_tensor handles case mismatch and skips None candidate values."""
        arr0 = np.array([0, 1])
        arr1 = np.array([[10, 20, 30, 40]])
        raw_tuple: Sequence[np.ndarray] = (arr0, arr1)

        # Case-insensitive match when dictionary has uppercase key and candidate is lowercase
        d_upper = {"Labels": arr0}
        self.assertIs(_find_output_tensor(d_upper, raw_tuple, ["labels"], 1), arr0)

        # Case-insensitive match when dictionary has lowercase key and candidate is uppercase
        d_lower = {"labels": arr0}
        self.assertIs(_find_output_tensor(d_lower, raw_tuple, ["LABELS"], 1), arr0)

        # Skip None candidate and fall back to next candidate or positional index
        d_none = {"labels": None, "pred_labels": arr0}
        self.assertIs(_find_output_tensor(d_none, raw_tuple, ["labels", "pred_labels"], 1), arr0)

        d_only_none = {"labels": None}
        self.assertIs(_find_output_tensor(d_only_none, raw_tuple, ["labels"], 1), arr1)

    def test_find_output_tensor_empty_and_negative_indices(self) -> None:
        """Verify _find_output_tensor error handling for empty sequences and negative indices."""
        arr0 = np.array([0, 1])

        # Negative fallback_index defaults to index 0 safely
        self.assertIs(_find_output_tensor({}, (arr0,), ["labels"], -1), arr0)

        # Empty sequence with no candidate match raises ValueError
        with self.assertRaises(ValueError):
            _find_output_tensor({}, (), ["labels"], 0)

    def test_predict_slice_zero_size_image(self) -> None:
        """Verify predict_slice gracefully returns empty detections for zero-dimension image."""
        mock_sess = MockSession()
        model = RTDetrV4ONNXModel(session=mock_sess, input_size=640)

        # PIL Image with 0 width
        zero_w_img = Image.new("RGB", (0, 640))
        self.assertEqual(model.predict_slice(zero_w_img), [])

        # PIL Image with 0 height
        zero_h_img = Image.new("RGB", (640, 0))
        self.assertEqual(model.predict_slice(zero_h_img), [])

    def test_duck_typed_session_without_input_output_methods(self) -> None:
        """Verify RTDetrV4ONNXModel operates with minimal duck-typed session missing get_inputs/get_outputs."""
        labels = np.array([[0]], dtype=np.int64)
        boxes = np.array([[[10.0, 10.0, 50.0, 50.0]]], dtype=np.float32)
        scores = np.array([[0.95]], dtype=np.float32)

        class MinimalSession:
            def run(self, output_names: Any, feed_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                return (labels, boxes, scores)

        minimal_sess = MinimalSession()
        model = RTDetrV4ONNXModel(session=minimal_sess, input_size=640)
        self.assertEqual(model._input_names, ["images"])
        self.assertEqual(model._output_names, [])

        img = Image.new("RGB", (640, 640))
        dets = model.predict_slice(img)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].bbox, (10.0, 10.0, 50.0, 50.0))


if __name__ == "__main__":
    unittest.main()
