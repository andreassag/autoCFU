"""Deterministic 2D grid image slicer for high-resolution colony plates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import SlicingConfig


@dataclass(frozen=True)
class SliceWindow:
    """Rectangular crop window in absolute image pixel coordinates.

    Attributes:
        xmin: Left edge coordinate (inclusive) in pixels.
        ymin: Top edge coordinate (inclusive) in pixels.
        xmax: Right edge coordinate (exclusive) in pixels.
        ymax: Bottom edge coordinate (exclusive) in pixels.
    """

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        """Tile width in pixels."""
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        """Tile height in pixels."""
        return self.ymax - self.ymin

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Window bounds formatted as ``(xmin, ymin, xmax, ymax)``."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)


class PlateSlicer:
    """Slices large images into a 2D overlapping grid of fixed-size tiles."""

    def __init__(
        self,
        slice_width: int = 640,
        slice_height: int = 640,
        overlap_width_ratio: float = 0.20,
        overlap_height_ratio: float = 0.20,
        *,
        overlap_ratio: float | None = None,
    ) -> None:
        """Initialize the 2D sliding-window tile generator.

        Args:
            slice_width: Width of each crop tile in pixels.
            slice_height: Height of each crop tile in pixels.
            overlap_width_ratio: Fractional horizontal overlap in ``[0.0, 1.0)``.
            overlap_height_ratio: Fractional vertical overlap in ``[0.0, 1.0)``.
            overlap_ratio: Optional symmetric overlap override applied to both axes.

        Raises:
            ValueError: If tile dimensions are non-positive or overlap ratios lie
                outside ``[0.0, 1.0)``.
        """
        if overlap_ratio is not None:
            overlap_width_ratio = overlap_ratio
            overlap_height_ratio = overlap_ratio
        if slice_width <= 0 or slice_height <= 0:
            raise ValueError("Slice dimensions must be positive integers.")
        if not (0.0 <= overlap_width_ratio < 1.0) or not (0.0 <= overlap_height_ratio < 1.0):
            raise ValueError("Overlap ratios must be in the range [0.0, 1.0).")

        self.slice_width = slice_width
        self.slice_height = slice_height
        self.overlap_width_ratio = overlap_width_ratio
        self.overlap_height_ratio = overlap_height_ratio

        self.step_x = max(1, int(slice_width * (1.0 - overlap_width_ratio)))
        self.step_y = max(1, int(slice_height * (1.0 - overlap_height_ratio)))

    @classmethod
    def from_config(cls, config: SlicingConfig) -> PlateSlicer:
        """Construct a :class:`PlateSlicer` from a :class:`SlicingConfig` instance."""
        return cls(
            slice_width=config.slice_width,
            slice_height=config.slice_height,
            overlap_width_ratio=config.overlap_width_ratio,
            overlap_height_ratio=config.overlap_height_ratio,
        )

    def _generate_1d_intervals(self, full_length: int, slice_len: int, step: int) -> list[tuple[int, int]]:
        """Compute start and end coordinates along one dimension."""
        if full_length <= slice_len:
            return [(0, full_length)]

        intervals: list[tuple[int, int]] = []
        start = 0
        while start + slice_len <= full_length:
            intervals.append((start, start + slice_len))
            start += step

        # Ensure the final boundary is covered by adding a right/bottom-anchored slice
        last_start = full_length - slice_len
        if intervals[-1][0] < last_start:
            intervals.append((last_start, full_length))

        return intervals

    def get_slice_windows(self, image_width: int, image_height: int) -> list[SliceWindow]:
        """Compute all 2D slice windows covering the given image dimensions.

        Args:
            image_width: Full image width in pixels.
            image_height: Full image height in pixels.

        Returns:
            Row-major list of :class:`SliceWindow` crop bounds.

        Raises:
            ValueError: If ``image_width`` or ``image_height`` is non-positive.
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive.")

        x_intervals = self._generate_1d_intervals(image_width, self.slice_width, self.step_x)
        y_intervals = self._generate_1d_intervals(image_height, self.slice_height, self.step_y)

        windows: list[SliceWindow] = []
        for ymin, ymax in y_intervals:
            for xmin, xmax in x_intervals:
                windows.append(SliceWindow(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))
        return windows

    def slice_image(
        self,
        image: np.ndarray | Image.Image,
    ) -> list[tuple[SliceWindow, np.ndarray]]:
        """Slice an image into cropped RGB NumPy tiles paired with window bounds.

        Args:
            image: Input dish image as a PIL image or RGB NumPy array.

        Returns:
            List of ``(SliceWindow, tile_array)`` tuples in row-major order.

        Raises:
            TypeError: If ``image`` is neither a :class:`PIL.Image.Image` nor a
                :class:`numpy.ndarray`.
        """
        if isinstance(image, Image.Image):
            width, height = image.size
            img_np = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            img_np = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL Image or numpy array.")

        windows = self.get_slice_windows(width, height)
        results: list[tuple[SliceWindow, np.ndarray]] = []
        for w in windows:
            crop = img_np[w.ymin : w.ymax, w.xmin : w.xmax]
            results.append((w, crop))
        return results
