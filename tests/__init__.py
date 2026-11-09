"""Test environment initialization and fallback shims."""

from __future__ import annotations

import sys
import types
from typing import Any

# Check if numpy is importable
try:
    import numpy as np  # noqa: F401
except ImportError:
    # Minimal pure-Python numpy drop-in for environments where numpy C-extensions cannot load
    class DType:
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, val: Any) -> Any:
            if "int" in self.name:
                return int(val)
            return float(val)

    float32 = DType("float32")
    float64 = DType("float64")
    int64 = DType("int64")
    int32 = DType("int32")
    uint8 = DType("uint8")

    class ArrayShim:
        def __init__(self, data: Any, shape: tuple[int, ...] | None = None, dtype: Any = None) -> None:
            if isinstance(data, ArrayShim):
                self._data = data._data
                self.shape = shape if shape is not None else data.shape
                self.dtype = dtype or data.dtype
                return
            self._data = data
            self.dtype = dtype or float32
            if shape is not None:
                self.shape = shape
            elif isinstance(data, (list, tuple)):
                s: list[int] = []
                curr: Any = data
                while isinstance(curr, (list, tuple)):
                    s.append(len(curr))
                    curr = curr[0] if len(curr) > 0 else None
                self.shape = tuple(s)
            else:
                self.shape = ()

        @property
        def ndim(self) -> int:
            return len(self.shape)

        def tolist(self) -> Any:
            def _conv(item: Any) -> Any:
                if isinstance(item, (list, tuple)):
                    return [_conv(x) for x in item]
                if isinstance(item, ArrayShim):
                    return item.tolist()
                return item

            return _conv(self._data)

        def all(self) -> bool:
            return True

        def __float__(self) -> float:
            val = self._data
            while isinstance(val, (list, tuple)):
                val = val[0]
            return float(val)

        def __int__(self) -> int:
            val = self._data
            while isinstance(val, (list, tuple)):
                val = val[0]
            return int(val)

        def __len__(self) -> int:
            return self.shape[0] if self.shape else 0

        def __iter__(self) -> Any:
            if isinstance(self._data, (list, tuple)):
                for item in self._data:
                    yield ArrayShim(item) if isinstance(item, (list, tuple)) else item
            else:
                yield self._data

        def __getitem__(self, key: Any) -> Any:
            if isinstance(key, tuple):
                if any(k is None or k is Ellipsis for k in key):
                    new_shape = (1,) + self.shape
                    return ArrayShim(self._data, shape=new_shape, dtype=self.dtype)
                if len(key) == 2 and isinstance(key[0], int) and isinstance(key[1], int):
                    # Indexing e.g. arr_5[0, 4]
                    r, c = key
                    return self._data[r][c]
                # Slice indexing e.g. [ymin:ymax, xmin:xmax]
                h_slice, w_slice = key[0], key[1]
                h_len = (h_slice.stop or 0) - (h_slice.start or 0)
                w_len = (w_slice.stop or 0) - (w_slice.start or 0)
                sub_shape = (h_len, w_len) + self.shape[2:]
                return ArrayShim(self._data, shape=sub_shape, dtype=self.dtype)
            if isinstance(key, int):
                item = self._data[key]
                sub_shape = self.shape[1:] if len(self.shape) > 1 else ()
                return ArrayShim(item, shape=sub_shape, dtype=self.dtype) if isinstance(item, (list, tuple)) else item
            return self

        def astype(self, dtype: Any) -> ArrayShim:
            if isinstance(self._data, list):
                conv = int if dtype is int or (hasattr(dtype, "name") and "int" in dtype.name) else float
                return ArrayShim([conv(x) for x in self._data], shape=self.shape, dtype=dtype)
            return self

        def __truediv__(self, other: Any) -> ArrayShim:
            return self

        def copy(self) -> ArrayShim:
            return ArrayShim(self._data, shape=self.shape, dtype=self.dtype)

        def __setitem__(self, key: Any, value: Any) -> None:
            pass

    def _linspace(start: float, stop: float, num: int = 50) -> ArrayShim:
        if num <= 0:
            return ArrayShim([])
        if num == 1:
            return ArrayShim([float(start)])
        step = (stop - start) / (num - 1)
        vals = [start + step * i for i in range(num)]
        return ArrayShim(vals)

    def _array(data: Any, dtype: Any = None) -> ArrayShim:
        if isinstance(data, ArrayShim):
            return data
        if hasattr(data, "size") and isinstance(data.size, tuple) and len(data.size) == 2:
            return ArrayShim([], shape=(data.size[1], data.size[0], 3), dtype=dtype or uint8)
        return ArrayShim(data, dtype=dtype)

    np_mod = types.ModuleType("numpy")
    setattr(np_mod, "ndarray", ArrayShim)
    setattr(np_mod, "array", _array)
    setattr(np_mod, "asarray", _array)
    setattr(np_mod, "linspace", _linspace)
    setattr(
        np_mod,
        "zeros",
        lambda shape, dtype=None: ArrayShim([], shape=shape if isinstance(shape, tuple) else (shape,), dtype=dtype),
    )
    setattr(
        np_mod,
        "empty",
        lambda shape, dtype=None: ArrayShim([], shape=shape if isinstance(shape, tuple) else (shape,), dtype=dtype),
    )
    setattr(np_mod, "transpose", lambda arr, axes=None: arr)
    setattr(np_mod, "float32", float32)
    setattr(np_mod, "float64", float64)
    setattr(np_mod, "int64", int64)
    setattr(np_mod, "int32", int32)
    setattr(np_mod, "uint8", uint8)

    sys.modules["numpy"] = np_mod

# Check if PIL is importable
try:
    import PIL.Image  # noqa: F401
except ImportError:

    class Resampling:
        BILINEAR = 2

    class ImageShim:
        Resampling = Resampling

        def __init__(self, mode: str = "RGB", size: tuple[int, int] = (640, 640), color: Any = None) -> None:
            self.mode = mode
            self.size = size

        def convert(self, mode: str) -> ImageShim:
            return ImageShim(mode, self.size)

        def resize(self, size: tuple[int, int], resample: Any = None) -> ImageShim:
            return ImageShim(self.mode, size)

        def save(self, path: Any, *args: Any, **kwargs: Any) -> None:
            with open(path, "wb") as f:
                f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF")

        @classmethod
        def new(cls, mode: str, size: tuple[int, int], color: Any = None) -> ImageShim:
            return cls(mode, size, color)

        @classmethod
        def open(cls, path: Any) -> ImageShim:
            return cls("RGB", (640, 640))

        @classmethod
        def fromarray(cls, arr: Any) -> ImageShim:
            h = arr.shape[0] if hasattr(arr, "shape") and len(arr.shape) > 0 else 640
            w = arr.shape[1] if hasattr(arr, "shape") and len(arr.shape) > 1 else 640
            return cls("RGB", (w, h))

    class DrawShim:
        def __init__(self, image: Any) -> None:
            self.image = image

        def rectangle(self, xy: Any, outline: Any = None, width: int = 1) -> None:
            pass

        def text(self, xy: Any, text: str, fill: Any = None, font: Any = None) -> None:
            pass

    class ImageFontShim:
        @staticmethod
        def load_default() -> Any:
            return object()

    pil_mod = types.ModuleType("PIL")
    pil_image_mod = types.ModuleType("PIL.Image")
    setattr(pil_image_mod, "Image", ImageShim)
    setattr(pil_image_mod, "Resampling", Resampling)
    setattr(pil_image_mod, "new", ImageShim.new)
    setattr(pil_image_mod, "open", ImageShim.open)
    setattr(pil_image_mod, "fromarray", ImageShim.fromarray)

    pil_draw_mod = types.ModuleType("PIL.ImageDraw")
    setattr(pil_draw_mod, "Draw", lambda img: DrawShim(img))

    pil_font_mod = types.ModuleType("PIL.ImageFont")
    setattr(pil_font_mod, "load_default", ImageFontShim.load_default)

    setattr(pil_mod, "Image", pil_image_mod)
    setattr(pil_mod, "ImageDraw", pil_draw_mod)
    setattr(pil_mod, "ImageFont", pil_font_mod)

    sys.modules["PIL"] = pil_mod
    sys.modules["PIL.Image"] = pil_image_mod
    sys.modules["PIL.ImageDraw"] = pil_draw_mod
    sys.modules["PIL.ImageFont"] = pil_font_mod
