"""Unit tests for PlateSlicer and SliceWindow."""

from __future__ import annotations

import tests  # noqa: F401  # isort: skip

import unittest

import numpy as np
from PIL import Image

from autocfu.config import SlicingConfig
from autocfu.slicer import PlateSlicer


class TestPlateSlicer(unittest.TestCase):
    def test_default_4000x4000_grid(self) -> None:
        """Test standard full-plate dimensions (4000x4000) with 640x640 tiles and 20% overlap."""
        slicer = PlateSlicer(
            slice_width=640,
            slice_height=640,
            overlap_width_ratio=0.20,
            overlap_height_ratio=0.20,
        )
        self.assertEqual(slicer.step_x, 512)
        self.assertEqual(slicer.step_y, 512)

        windows = slicer.get_slice_windows(4000, 4000)

        # 8 horizontal windows x 8 vertical windows = 64 slices
        self.assertEqual(len(windows), 64)

        # First window should be at top-left
        first = windows[0]
        self.assertEqual(first.bbox, (0, 0, 640, 640))

        # Last window should reach bottom-right corner exactly
        last = windows[-1]
        self.assertEqual(last.bbox, (3360, 3360, 4000, 4000))
        self.assertEqual(last.xmax, 4000)
        self.assertEqual(last.ymax, 4000)

        # All windows should have standard tile dimensions
        for w in windows:
            self.assertEqual(w.width, 640)
            self.assertEqual(w.height, 640)

    def test_pixel_coverage(self) -> None:
        """Ensure every coordinate in the image is covered by at least one slice."""
        width, height = 1250, 950
        slicer = PlateSlicer(slice_width=400, slice_height=300, overlap_width_ratio=0.25, overlap_height_ratio=0.25)
        windows = slicer.get_slice_windows(width, height)

        coverage_mask = np.zeros((height, width), dtype=bool)
        for w in windows:
            coverage_mask[w.ymin : w.ymax, w.xmin : w.xmax] = True

        self.assertTrue(coverage_mask.all(), "All pixels must be covered by generated slice windows.")

    def test_image_smaller_than_slice(self) -> None:
        """When the input image is smaller than slice dimensions, return a single covering slice."""
        slicer = PlateSlicer(slice_width=640, slice_height=640)
        windows = slicer.get_slice_windows(300, 400)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].bbox, (0, 0, 300, 400))

    def test_exact_tile_size(self) -> None:
        """When input matches slice size exactly, return 1 slice."""
        slicer = PlateSlicer(slice_width=640, slice_height=640)
        windows = slicer.get_slice_windows(640, 640)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].bbox, (0, 0, 640, 640))

    def test_zero_overlap(self) -> None:
        """Zero overlap should tile strictly by slice dimension."""
        slicer = PlateSlicer(slice_width=200, slice_height=200, overlap_width_ratio=0.0, overlap_height_ratio=0.0)
        self.assertEqual(slicer.step_x, 200)
        self.assertEqual(slicer.step_y, 200)
        windows = slicer.get_slice_windows(400, 400)
        self.assertEqual(len(windows), 4)

    def test_floor_step_calculation(self) -> None:
        """Verify step size strictly follows floor formula int(size * (1 - overlap))."""
        # 640 * (1 - 0.33) = 640 * 0.67 = 428.8 -> floor = 428
        slicer = PlateSlicer(slice_width=640, slice_height=640, overlap_width_ratio=0.33, overlap_height_ratio=0.33)
        self.assertEqual(slicer.step_x, 428)
        self.assertEqual(slicer.step_y, 428)

    def test_overlap_ratio_alias(self) -> None:
        """Verify overlap_ratio keyword argument sets both width and height overlap ratios."""
        slicer = PlateSlicer(slice_width=640, slice_height=640, overlap_ratio=0.25)
        self.assertEqual(slicer.overlap_width_ratio, 0.25)
        self.assertEqual(slicer.overlap_height_ratio, 0.25)
        self.assertEqual(slicer.step_x, 480)
        self.assertEqual(slicer.step_y, 480)

    def test_invalid_parameters(self) -> None:
        """Verify parameter validation for slice sizes and overlap ratios."""
        with self.assertRaises(ValueError):
            PlateSlicer(slice_width=-10)
        with self.assertRaises(ValueError):
            PlateSlicer(slice_height=0)
        with self.assertRaises(ValueError):
            PlateSlicer(overlap_width_ratio=1.0)
        with self.assertRaises(ValueError):
            PlateSlicer(overlap_height_ratio=-0.1)

    def test_slice_image_numpy_and_pil(self) -> None:
        """Test slice_image function with both PIL and numpy array inputs."""
        slicer = PlateSlicer(slice_width=100, slice_height=100, overlap_width_ratio=0.2)
        np_img = np.zeros((200, 200, 3), dtype=np.uint8)

        # Slice numpy array
        np_slices = slicer.slice_image(np_img)
        self.assertGreater(len(np_slices), 0)
        for _win, crop in np_slices:
            self.assertEqual(crop.shape, (100, 100, 3))

        # Slice PIL image
        pil_img = Image.fromarray(np_img)
        pil_slices = slicer.slice_image(pil_img)
        self.assertEqual(len(np_slices), len(pil_slices))
        for _win, crop in pil_slices:
            self.assertEqual(crop.shape, (100, 100, 3))

    def test_from_config(self) -> None:
        """Verify initialization via SlicingConfig."""
        cfg = SlicingConfig(slice_width=512, slice_height=512, overlap_width_ratio=0.15, overlap_height_ratio=0.15)
        slicer = PlateSlicer.from_config(cfg)
        self.assertEqual(slicer.slice_width, 512)
        self.assertAlmostEqual(slicer.overlap_width_ratio, 0.15)


if __name__ == "__main__":
    unittest.main()
