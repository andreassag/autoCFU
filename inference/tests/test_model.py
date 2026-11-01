"""Unit tests for RTDetrV2ONNXModel session wrapper and provider resolution."""

import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from autocfu.model import RTDetrV2ONNXModel


class MockNode:
    def __init__(self, name: str) -> None:
        self.name = name


class MockSession:
    """Mock ONNX Runtime session returning deterministic output tensors."""

    def __init__(
        self,
        input_names: List[str] = ("images", "orig_target_sizes"),
        output_names: List[str] = ("labels", "boxes", "scores"),
        outputs: Optional[List[np.ndarray]] = None,
    ) -> None:
        self.input_nodes = [MockNode(n) for n in input_names]
        self.output_nodes = [MockNode(n) for n in output_names]
        self.last_feed_dict: Dict[str, Any] = {}

        if outputs is not None:
            self.outputs = outputs
        else:
            # Default: 1 batch, 2 detections
            labels = np.array([[0, 1]], dtype=np.int64)
            boxes = np.array([[[10.0, 15.0, 50.0, 55.0], [100.0, 100.0, 150.0, 150.0]]], dtype=np.float32)
            scores = np.array([[0.92, 0.45]], dtype=np.float32)
            self.outputs = [labels, boxes, scores]

    def get_inputs(self) -> List[MockNode]:
        return self.input_nodes

    def get_outputs(self) -> List[MockNode]:
        return self.output_nodes

    def get_providers(self) -> List[str]:
        return ["CPUExecutionProvider"]

    def run(self, output_names: Any, feed_dict: Dict[str, Any]) -> List[np.ndarray]:
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

    @unittest.skipIf(ort is None, "onnxruntime is not installed")
    def test_real_onnx_model_loading(self) -> None:
        """Test loading a real physical ONNX model file from disk."""
        bundled_onnx = Path(ort.__file__).parent / "datasets" / "mul_1.onnx"
        if bundled_onnx.is_file():
            model = RTDetrV2ONNXModel(model_path=bundled_onnx, device="cpu")
            self.assertEqual(model.active_provider, "CPUExecutionProvider")
            self.assertIn("X", model._input_names)


if __name__ == "__main__":
    unittest.main()
