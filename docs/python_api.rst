Python API & Examples
=====================

AutoCFU provides an intuitive, typed Python library for integrating sliced microbial colony detection into automated lab equipment, analysis scripts, and computer vision pipelines.

Quickstart
----------

Initialize an :class:`~autocfu.pipeline.AutoCFUEngine` with a model path and run sliced inference on a high-resolution agar plate:

.. code-block:: python

   from autocfu import AutoCFUEngine

   # Initialize engine with an exported ONNX model
   engine = AutoCFUEngine(
       model_path="models/rtdetrv4_colony_detector.onnx",
       device="auto",  # Uses CUDA if available, falls back to CPU
   )

   # Run sliced inference on a high-resolution Petri dish
   result = engine.infer_image("data/dish_4000x4000.jpg")

   print(f"Detected {result.colony_count} colonies across {result.num_slices} tiles.")
   print(f"Inference duration: {result.execution_time_seconds:.2f} seconds.")

Working with InferenceResult
----------------------------

The :class:`~autocfu.pipeline.InferenceResult` dataclass encapsulates all detections, global dish coordinates, and benchmark performance metrics.

Accessing Detection Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Total colony enumeration
   print(f"Colony Count: {result.colony_count}")

   # Bounding boxes as list of tuples [(x1, y1, x2, y2), ...]
   for bbox in result.bboxes[:5]:
       x1, y1, x2, y2 = bbox
       print(f"Colony Box: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")

   # Colony centroids as list of tuples [(cx, cy), ...]
   for cx, cy in result.centers[:5]:
       print(f"Centroid: ({cx:.1f}, {cy:.1f})")

   # Detection confidence scores
   for score in result.scores[:5]:
       print(f"Confidence: {score:.3f}")

NumPy Array Exports
^^^^^^^^^^^^^^^^^^^

Convert bounding boxes and centroids directly into NumPy arrays for downstream spatial analysis or machine learning:

.. code-block:: python

   # Export [N, 5] array: columns are [x1, y1, x2, y2, score]
   boxes_and_scores = result.to_numpy(include_score=True)
   print("Shape:", boxes_and_scores.shape)  # e.g., (247, 5)

   # Export [N, 4] bounding boxes only
   boxes_only = result.to_numpy(include_score=False)

   # Export [N, 2] colony centroid coordinates [cx, cy]
   centroids = result.centers_numpy()

Structured File Exports
^^^^^^^^^^^^^^^^^^^^^^^

Export detections to JSON or CSV formats:

.. code-block:: python

   # Save structured JSON
   result.save_json("results/detections.json", indent=2)

   # Save CSV detection table
   result.save_csv("results/detections.csv")

Rendering Visual Overlays
^^^^^^^^^^^^^^^^^^^^^^^^^

Generate annotated images with colored bounding boxes and confidence labels:

.. code-block:: python

   # Draw overlays and save directly to file
   result.draw_overlay(
       output_path="results/overlay.jpg",
       box_color=(0, 255, 0),  # Green boxes
       box_width=2,
       show_score=True,
   )

   # Or render to an in-memory PIL Image object
   annotated_pil_image = result.draw_overlay()
   annotated_pil_image.show()

Spatial Filtering: ROIs & Circular Zones
----------------------------------------

In microbiological assays (such as spot plating, spiral plating, or disk diffusion antibiotic susceptibility testing), colonies may only need to be enumerated within specific spatial regions.

Rectangular Region of Interest (ROI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Filter colonies located within a specific rectangular bounding box:

.. code-block:: python

   # Filter colonies whose centroid falls inside [x1, y1, x2, y2]
   roi_colonies = result.filter_by_roi(
       x1=500.0,
       y1=500.0,
       x2=2500.0,
       y2=2500.0,
       by_center=True,  # Default: match based on colony centroid
   )
   print(f"Colonies inside ROI: {len(roi_colonies)}")

Circular Drop Spot or Inhibition Zone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Filter colonies whose centroid falls within a circular zone (e.g., pipetted droplet spot or antibiotic susceptibility clearance zone):

.. code-block:: python

   # Center at (2000, 2000) with radius 400 pixels
   spot_colonies = result.filter_by_circle(cx=2000.0, cy=2000.0, radius=400.0)
   print(f"Colonies inside spot: {len(spot_colonies)}")

Batch Processing & Aggregated Summaries
---------------------------------------

Process multiple Petri dish images sequentially and export a consolidated batch summary:

.. code-block:: python

   from pathlib import Path
   from autocfu import AutoCFUEngine, export_batch_summary_csv

   engine = AutoCFUEngine(model_path="models/rtdetrv4_colony_detector.onnx")

   # Collect plate paths
   plate_dir = Path("data/agar_plates/")
   image_paths = sorted(plate_dir.glob("*.jpg"))

   # Run batch inference
   batch_results = engine.infer_batch(image_paths)

   # Export consolidated summary CSV
   export_batch_summary_csv(batch_results, "results/batch_summary.csv")

In-Memory Image Inputs
----------------------

In automated robotic imaging systems or streaming video pipelines, images are often kept in memory. :class:`~autocfu.pipeline.AutoCFUEngine` accepts `PIL.Image.Image` objects or `numpy.ndarray` RGB matrices:

.. code-block:: python

   from PIL import Image
   import numpy as np

   # From a PIL Image
   pil_image = Image.open("data/dish.jpg").convert("RGB")
   res_pil = engine.infer_image(pil_image)

   # From a NumPy RGB array (e.g. from OpenCV or camera buffer)
   # Note: OpenCV loads BGR by default; ensure it is converted to RGB
   rgb_array = np.array(pil_image)
   res_np = engine.infer_image(rgb_array)

Programmatic Configuration
--------------------------

Fine-tune tile slicing dimensions, overlap ratios, confidence thresholds, and merging algorithms programmatically:

.. code-block:: python

   from autocfu import EngineConfig, InferenceConfig, SlicingConfig, AutoCFUEngine

   # Create custom configuration
   config = EngineConfig(
       slicing=SlicingConfig(
           slice_width=640,
           slice_height=640,
           overlap_width_ratio=0.25,  # 25% overlap
           overlap_height_ratio=0.25,
       ),
       inference=InferenceConfig(
           confidence_threshold=0.40,
           merger_type="greedy_nmm",  # "greedy_nmm" or "nms"
           match_metric="ios",        # "iou" or "ios"
           postprocess_match_threshold=0.50,
           intra_tile_nms_threshold=0.45,
           min_box_size=6.0,          # Filter sub-pixel sensor noise
           max_box_size=300.0,        # Filter dish rim reflections
           max_aspect_ratio=2.5,      # Filter elongated border artifacts
       ),
   )

   # Initialize engine with the custom config
   engine = AutoCFUEngine(
       model_path="models/rtdetrv4_colony_detector.onnx",
       config=config,
   )

Low-Level Component Usage
-------------------------

AutoCFU's internal components can also be used independently.

PlateSlicer (Grid Generation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate overlapping tiles and track pixel offsets:

.. code-block:: python

   from PIL import Image
   from autocfu import PlateSlicer

   slicer = PlateSlicer(slice_width=640, slice_height=640, overlap_ratio=0.20)
   image = Image.open("data/dish_4000x4000.jpg")

   for window, crop in slicer.slice_image(image):
       print(f"Tile at ({window.xmin}, {window.ymin}): crop shape {crop.shape}")

GreedyNMM / NMS (Detection Merging)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Merge raw detections across tile boundaries using confidence-weighted centroid merging:

.. code-block:: python

   from autocfu import Detection, GreedyNMM

   raw_detections = [
       Detection(bbox=(100.0, 100.0, 130.0, 130.0), score=0.92),
       Detection(bbox=(102.0, 101.0, 132.0, 131.0), score=0.88),  # Overlapping detection from adjacent tile
   ]

   # Accepts iou_threshold (or match_threshold alias)
   merger = GreedyNMM(iou_threshold=0.5, match_metric="iou")
   merged = merger.merge(raw_detections)

   print(f"Merged {len(raw_detections)} detections into {len(merged)} colony.")
   print("Merged Box:", merged[0].bbox)
