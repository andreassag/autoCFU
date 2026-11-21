AutoCFU Documentation
=====================

**AutoCFU** is a standalone, dependency-minimal inference engine implemented in both
**Python** and **Rust** for automated microbial colony detection, star-convex
instance segmentation, and plate/section countability classification using exported
ONNX Runtime models.

Supported Permissive Model Architectures
----------------------------------------

1. **Radial CenterNet** (``centernet``) — Keypoint center localization ``(cx, cy)`` and
   colony radius estimation ``r_px``.
2. **2D StarDist** (``stardist``) — 32-ray star-convex polygon instance segmentation
   with 16-bit ``uint16`` instance mask export.
3. **Plate & Section Countability Classifier** (``countability``) — 3-class
   classification (``empty``, ``countable``, ``uncountable``) for full plates and
   ``640×640`` sections.

.. note::
   All source code, mathematical pipelines, and supported architectures in AutoCFU
   are permissively licensed (**Apache-2.0**, **MIT**, **BSD-3-Clause**) and cleared
   for commercial use with proprietary or permissively licensed ONNX weights.

.. toctree::
   :maxdepth: 2
   :caption: Guides & Reference

   installation
   models
   cli
   python_api
   rust_api
