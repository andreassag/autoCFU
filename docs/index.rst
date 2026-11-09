AutoCFU Documentation
=====================

**AutoCFU** is a lightweight, high-performance sliced inference engine and command-line utility for automated detection, segmentation, and enumeration of microbial colonies on Petri dishes.

Overview & Architecture
-----------------------

High-resolution Petri dish photography typically captures agar plates at 4000×4000 pixels. Directly downscaling a 4000×4000 image to standard detector input dimensions (such as 640×640) degrades minute microbial colonies (<15 pixels in diameter) to unrecognizable sub-pixel noise.

To preserve microscopic colony morphology without requiring high-memory GPUs, AutoCFU implements a **Slicing-Aided Hyper Inference (SAHI)** and **Greedy Non-Maximum Merging (GreedyNMM)** pipeline:

1. **Grid Generation (`autocfu.slicer`)**:
   Slices the input dish into overlapping tiles of fixed dimensions (default: 640×640) with configured overlap (default: 20%). Step stride is calculated as:

   .. math::

      \text{step} = \lfloor \text{slice\_size} \times (1.0 - \text{overlap}) \rfloor

   Boundary edges are clamped to guarantee full image coverage without empty padding artifacts.

2. **Batch ONNX Inference (`autocfu.model`)**:
   Runs each image slice through an exported RT-DETRv4 ONNX model using ONNX Runtime. The inference engine intentionally excludes heavyweight training dependencies (like PyTorch or CUDA toolchains) from runtime dependencies to keep installation lightweight and portable.

3. **Global Coordinate Mapping (`autocfu.pipeline`)**:
   Translates bounding box coordinates from slice-local coordinates :math:`(x_1, y_1, x_2, y_2)` back to global dish coordinates:

   .. math::

      (x_1 + x_{\text{offset}}, y_1 + y_{\text{offset}}, x_2 + x_{\text{offset}}, y_2 + y_{\text{offset}})

4. **Greedy Non-Maximum Merging (`autocfu.merger`)**:
   Merges redundant detections across overlapping tile boundaries using GreedyNMM or standard NMS, clustering intersecting bounding boxes and computing confidence-weighted centroid coordinates:

   .. math::

      \mathbf{box}_{\text{merged}} = \frac{\sum_{i} \text{score}_i \times \mathbf{box}_i}{\sum_{i} \text{score}_i}

ONNX Tensor Contract
--------------------

AutoCFU expects models exported using the RT-DETRv4 ONNX export pipeline:

* **Inputs**:
  - ``images``: ``[batch_size, 3, 640, 640]``, float32, normalized to ``[0.0, 1.0]`` (RGB, NCHW).
  - ``orig_target_sizes``: ``[batch_size, 2]``, int64, original slice dimensions (``[[640, 640]]``).

* **Outputs**:
  - ``labels`` (or ``pred_labels``, ``classes``): ``[batch_size, num_queries]``, int64 class IDs (``0`` = colony).
  - ``boxes`` (or ``pred_boxes``, ``bboxes``): ``[batch_size, num_queries, 4]``, float32 pixel coordinates ``[x1, y1, x2, y2]``.
  - ``scores`` (or ``pred_scores``, ``confs``): ``[batch_size, num_queries]``, float32 confidence scores in ``[0.0, 1.0]``.

User Guide & Examples
---------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide & Examples

   installation
   cli
   python_api

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/config
   api/slicer
   api/merger
   api/model
   api/pipeline
   api/cli
   api/utils

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
