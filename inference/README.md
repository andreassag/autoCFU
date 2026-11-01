# AutoCFU: Python Inference Engine (`autocfu`)

High-resolution sliced inference engine for automated microbial colony detection and enumeration on Petri dishes using fine-tuned **RT-DETRv2** and **Greedy Non-Maximum Merging (GreedyNMM)**.

---

## Overview & Architecture

High-resolution Petri dish / agar plate images (typically 4000×4000 pixels) cannot be directly fed into neural networks without significant downsampling, which destroys minute bacterial colonies (<15 pixels in diameter).

AutoCFU solves the resolution mismatch via Slicing-Aided Hyper Inference:

```
Full Dish (4000×4000)
       │
       ▼  PlateSlicer
640×640 Overlapping Tiles (20% overlap, step 512 px)
       │
       ▼  RTDetrV2ONNXModel (ONNX Runtime, CUDA/CPU)
Tile Predictions (Boxes, Scores, Labels)
       │
       ▼  Coordinate Translation ([x + x_offset, y + y_offset])
Global Coordinate Bounding Boxes
       │
       ▼  GreedyNMM (IoU threshold 0.50)
Merged Colony Count & Detections (Weighted centroids)
```

1. **`slicer.py` (`PlateSlicer`)**: Generates deterministic 2D sliding windows with configurable overlap and edge padding/clamping.
2. **`model.py` (`RTDetrV2ONNXModel`)**: High-performance ONNX Runtime session execution with automated provider discovery (CUDA GPU or CPU fallback).
3. **`merger.py` (`GreedyNMM`)**: Clusters overlapping detections from adjacent tiles and computes confidence-weighted coordinates to avoid double-counting boundary colonies.
4. **`pipeline.py` (`AutoCFUEngine`)**: End-to-end full plate runner returning structured `InferenceResult`.
5. **`cli.py` (`autocfu`)**: Production CLI for single images or directory batch jobs.

---

## Installation

### From the root repository
```bash
make install
```

### Direct pip installation
```bash
cd inference
pip install -e ".[dev]"
```

---

## Python API Usage

```python
from autocfu import AutoCFUEngine, load_config

# Load configuration from YAML or use defaults
config = load_config("configs/default_inference.yaml")

# Initialize engine
engine = AutoCFUEngine(model_path="models/rtdetrv2_r18vd_agar_640.onnx", config=config)

# Run full-plate inference
result = engine.infer_image("path/to/plate.jpg")

print(f"Colony Count: {result.colony_count}")
print(f"Processed {result.num_slices} tiles in {result.execution_time_seconds:.2f}s")

# Access bounding boxes
for det in result.detections:
    print(f"Box: {det.bbox}, Confidence: {det.score:.3f}")

# Export visual overlay and structured JSON/CSV
result.draw_overlay("results/plate_annotated.jpg")
result.save_csv("results/plate_summary.csv")
```

---

## Command-Line Interface (`autocfu`)

```bash
# Count colonies on a single plate and generate visual overlay
autocfu --image plate.jpg \
        --model models/rtdetrv2_r18vd_agar_640.onnx \
        --output-dir results/ \
        --save-overlay

# Batch process a directory of plates and write CSV summary
autocfu --input-dir /data/plates/ \
        --model models/rtdetrv2_r18vd_agar_640.onnx \
        --output-dir results/ \
        --save-csv \
        --save-json
```

*(Note: `autocfu-infer` and `agar-infer` are also available as command aliases).*

### CLI Options
- `--image`: Path to single plate image file.
- `--input-dir`: Directory containing images to process.
- `--model`: Path to exported RT-DETRv2 ONNX model.
- `--config`: Path to YAML configuration (default: `configs/default_inference.yaml` if present).
- `--confidence`: Detection score threshold (overrides config, default `0.35`).
- `--overlap`: Overlap ratio between adjacent tiles (overrides config, default `0.20`).
- `--slice-size`: Slice window width and height (overrides config, default `640`).
- `--output-dir`: Destination directory for outputs (default: `results`).
- `--save-overlay`: Render and save bounding box overlays on output images.
- `--save-csv`: Export a summary CSV with image filenames, counts, and runtimes.
- `--save-json`: Export structured JSON bounding box coordinates for each plate.
- `--merger`: Post-processing merger algorithm (`greedy_nmm` or `nms`).
- `--match-metric`: Bounding box metric (`iou` or `ios`).
- `--match-threshold`: Intersection threshold for merging boxes (default: `0.50`).
- `--device`: Execution device: `auto`, `cuda`, or `cpu`.

---

## Testing

Run unit tests via `make test` or `pytest`:
```bash
pytest tests/
```
