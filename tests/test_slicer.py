"""Unit tests for 2D sliding-window dish slicing."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from autocfu.config import SlicingConfig
from autocfu.slicer import PlateSlicer, SliceWindow


def test_single_tile_image_smaller_than_slice() -> None:
    slicer = PlateSlicer(slice_width=640, slice_height=640, overlap_ratio=0.20)
    windows = slicer.get_slice_windows(500, 400)
    assert windows == [SliceWindow(xmin=0, ymin=0, xmax=500, ymax=400)]


def test_edge_clamping_covers_1000x1000() -> None:
    slicer = PlateSlicer(slice_width=640, slice_height=640, overlap_ratio=0.20)
    assert slicer.step_x == 512
    assert slicer.step_y == 512
    windows = slicer.get_slice_windows(1000, 1000)
    assert windows == [
        SliceWindow(xmin=0, ymin=0, xmax=640, ymax=640),
        SliceWindow(xmin=360, ymin=0, xmax=1000, ymax=640),
        SliceWindow(xmin=0, ymin=360, xmax=640, ymax=1000),
        SliceWindow(xmin=360, ymin=360, xmax=1000, ymax=1000),
    ]


def test_highres_4000x4000_full_coverage() -> None:
    slicer = PlateSlicer.from_config(SlicingConfig())
    windows = slicer.get_slice_windows(4000, 4000)
    assert len(windows) == 64
    assert windows[0].xmin == 0 and windows[0].ymin == 0
    assert windows[-1].xmax == 4000 and windows[-1].ymax == 4000
    for win in windows:
        assert win.width == 640
        assert win.height == 640


def test_slice_image_array_and_pil() -> None:
    slicer = PlateSlicer(slice_width=64, slice_height=64, overlap_ratio=0.25)
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    tiles_arr = slicer.slice_image(arr)
    assert len(tiles_arr) == 4
    for _, tile in tiles_arr:
        assert tile.shape == (64, 64, 3)

    pil_img = Image.new("RGB", (100, 100), color=(12, 34, 56))
    tiles_pil = slicer.slice_image(pil_img)
    assert len(tiles_pil) == 4


def test_invalid_slicer_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        PlateSlicer(slice_width=0, slice_height=640)
    with pytest.raises(ValueError, match="Overlap"):
        PlateSlicer(overlap_width_ratio=1.0)
    slicer = PlateSlicer()
    with pytest.raises(ValueError, match="positive"):
        slicer.get_slice_windows(0, 640)
