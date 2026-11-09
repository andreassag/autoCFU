Command-Line Interface (CLI)
==============================

AutoCFU provides an intuitive and flexible command-line interface for running sliced colony inference on individual high-resolution Petri dish photographs or batch processing entire image folders.

CLI Entry Points
----------------

After installation, the CLI is available under three aliases:

* ``autocfu`` (primary command)
* ``autocfu-infer``
* ``agar-infer``

All aliases provide identical functionality and share the same arguments.

Basic Usage Examples
--------------------

Single-Image Inference with Visual Overlays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Process a single high-resolution dish photograph, count colonies, and save an annotated bounding box overlay:

.. code-block:: bash

   autocfu --image data/dish_4000x4000.jpg \
           --model models/rtdetrv4_colony_detector.onnx \
           --save-overlay \
           --output-dir results/

This will output:

.. code-block:: text

   Starting inference on 1 plate image(s)...
   Settings: slice=640x640, overlap=20%, conf_thresh=0.35, merger=greedy_nmm
   [1/1] dish_4000x4000.jpg: 247 colonies (49 tiles, 1.42s)

   Done. Processed 1 image(s), detected 247 colonies total.

The annotated image is saved to ``results/dish_4000x4000_detected.jpg``.

Batch Processing a Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Process all images in a directory and generate overlays, JSON coordinates, and per-plate and aggregated CSV reports:

.. code-block:: bash

   autocfu --input-dir /path/to/plates/ \
           --model models/rtdetrv4_colony_detector.onnx \
           --save-overlay \
           --save-json \
           --save-csv \
           --output-dir results/batch_run/

When multiple images are processed with ``--save-csv``, AutoCFU produces:
* Per-plate detection coordinates: ``results/batch_run/{stem}_detections.csv``
* Per-plate JSON coordinates: ``results/batch_run/{stem}_detections.json``
* Per-plate overlay image: ``results/batch_run/{stem}_detected.jpg``
* Aggregated summary CSV table: ``results/batch_run/summary.csv``

Customizing Slicing & Merging via CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can customize tile dimensions, overlap ratios, confidence thresholds, and merging algorithms directly from the CLI:

.. code-block:: bash

   autocfu --image dish.jpg \
           --model models/rtdetrv4_colony_detector.onnx \
           --slice-size 800 \
           --overlap 0.25 \
           --confidence 0.40 \
           --merger greedy_nmm \
           --match-metric ios \
           --match-threshold 0.50 \
           --intra-tile-threshold 0.45 \
           --intra-tile-metric ios \
           --min-box-size 6.0 \
           --max-box-size 250.0 \
           --max-aspect-ratio 2.5 \
           --device cuda \
           --save-overlay

Using Configuration Files
^^^^^^^^^^^^^^^^^^^^^^^^^

Specify a custom YAML configuration file instead of passing individual CLI flags:

.. code-block:: bash

   autocfu --image dish.jpg \
           --model models/rtdetrv4_colony_detector.onnx \
           --config configs/custom_inference.yaml \
           --save-overlay

If ``--config`` is omitted, AutoCFU automatically searches for ``configs/default_inference.yaml`` in the working directory.

CLI Arguments Reference
-----------------------

Input & Model Arguments
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Flag
     - Type
     - Description
   * - ``--image``
     - string
     - Path to a single agar plate image file (mutually exclusive with ``--input-dir``).
   * - ``--input-dir``
     - string
     - Path to a directory containing images for batch processing.
   * - ``--model``
     - string
     - Path to the exported RT-DETRv4 ONNX model file.
   * - ``--config``
     - string
     - Optional path to inference YAML configuration file.
   * - ``--device``
     - choice
     - ONNX Runtime execution provider: ``auto``, ``cuda``, or ``cpu`` (default: ``auto``).

Output Control Flags
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Flag
     - Type
     - Description
   * - ``--output-dir``
     - string
     - Output directory to save results and artifacts (default: ``results``).
   * - ``--save-overlay``
     - flag
     - Save annotated image with rendered colony bounding boxes and count banner.
   * - ``--save-json``
     - flag
     - Save structured JSON file with detection coordinates, scores, and metadata.
   * - ``--save-csv``
     - flag
     - Save CSV detection tables and an aggregated ``summary.csv`` for batch runs.

Tiling & Merger Overrides
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Flag
     - Type
     - Description
   * - ``--confidence``
     - float
     - Minimum detection score threshold (e.g. ``0.35``).
   * - ``--slice-size``
     - int
     - Width and height of square slice windows in pixels (e.g. ``640``).
   * - ``--overlap``
     - float
     - Overlap ratio between adjacent tiles (e.g. ``0.20`` for 20% overlap).
   * - ``--merger``
     - choice
     - Cross-tile duplicate merger: ``greedy_nmm`` or ``nms``.
   * - ``--match-metric``
     - choice
     - Overlap matching metric: ``iou`` (Intersection over Union) or ``ios`` (Intersection over Smaller).
   * - ``--match-threshold``
     - float
     - Clustering metric threshold for GreedyNMM / NMS (e.g. ``0.50``).
   * - ``--intra-tile-threshold``
     - float
     - Duplicate suppression threshold within individual tiles (e.g. ``0.45``, ``0.0`` to disable).
   * - ``--intra-tile-metric``
     - choice
     - Intra-tile overlap metric: ``iou`` or ``ios`` (default: ``ios``).
   * - ``--min-box-size``
     - float
     - Minimum colony box dimension in pixels (suppresses sensor noise, ``0.0`` to disable).
   * - ``--max-box-size``
     - float
     - Maximum colony box dimension in pixels (suppresses dish boundary reflections, ``0.0`` to disable).
   * - ``--max-aspect-ratio``
     - float
     - Maximum aspect ratio :math:`\max(w/h, h/w)` (suppresses border slivers, ``0.0`` to disable).

Output File Schemas
-------------------

Overlay Image (``*_detected.jpg``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An annotated JPEG containing:
* High-contrast bounding boxes around each detected colony.
* Numerical confidence score labels rendered adjacent to each detected colony.

JSON Coordinate Export (``*_detections.json``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "image_path": "data/dish_01.jpg",
     "image_width": 4000,
     "image_height": 4000,
     "colony_count": 247,
     "num_slices": 49,
     "execution_time_seconds": 1.423,
     "detections": [
       {
         "bbox": [1240.5, 830.2, 1265.1, 854.8],
         "center": [1252.8, 842.5],
         "score": 0.942,
         "class_id": 0,
         "label": "colony"
       }
     ]
   }

CSV Detection Table (``*_detections.csv``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   detection_index,class_id,label,score,x1,y1,x2,y2
   1,0,colony,0.9420,1240.50,830.20,1265.10,854.80
   2,0,colony,0.9150,2104.10,1452.00,2129.80,1478.40

Batch Summary Table (``summary.csv``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   image_path,width,height,colony_count,num_slices,time_sec
   data/dish_01.jpg,4000,4000,247,49,1.4230
   data/dish_02.jpg,4000,4000,189,49,1.3812
