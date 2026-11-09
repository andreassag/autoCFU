# AutoCFU: High-Resolution Microbial Colony Detection & Counting

An automated, high-resolution colony detection and enumeration system for microbiological Petri dishes using **RT-DETRv4** and **Slicing-Aided Hyper Inference (SAHI / GreedyNMM)**.

---

## Architecture & Repository Organization

Full-dish agar photography typically captures Petri dishes at 4000×4000 resolution. Direct downscaling reduces minute colonies (<15 pixels) to unrecognizable noise. This repository hosts the standalone, production-grade Python inference engine and CLI for sliced inference and automated colony enumeration.

```
autoCFU/
├── .github/
│   ├── dependabot.yml           # Automated weekly dependency updates (pip & GitHub Actions)
│   └── workflows/
│       ├── ci.yml               # Multi-version CI matrix (Python 3.10, 3.11, 3.12)
│       ├── dependabot-auto-merge.yml
│       └── docs.yml             # GitHub Pages documentation deployment workflow
│
├── docs/                        # Sphinx Documentation Suite (Furo theme)
│   ├── conf.py                  # Sphinx build configuration
│   ├── index.rst                # Master documentation landing page
│   ├── installation.rst         # Installation, setup, & ONNX provider guide
│   ├── cli.rst                  # CLI user guide & argument reference
│   ├── python_api.rst           # Python API examples & spatial filtering
│   └── api/                     # Autodoc module reference pages
│
├── src/autocfu/                 # Standalone Python Inference Engine
│   ├── __init__.py              # Public exports
│   ├── config.py                # Dataclass schemas and YAML loader
│   ├── slicer.py                # 2D Grid Window Tiler (step = size * (1 - overlap))
│   ├── merger.py                # Greedy Non-Maximum Merging (GreedyNMM & NMS)
│   ├── model.py                 # ONNX Runtime session manager (CUDA & CPU)
│   ├── pipeline.py              # Full-plate end-to-end sliced inference orchestrator
│   ├── utils.py                 # Ground-truth extraction, streaming & visualization helpers
│   └── cli.py                   # autocfu CLI entrypoint
│
├── tests/                       # Unit Test Suite
│   ├── test_slicer.py           # Slicer grid offsets & boundary tests
│   ├── test_merger.py           # IoU and weighted coordinate clustering tests
│   ├── test_model.py            # ONNX session unbatching & provider tests
│   ├── test_pipeline.py         # Pipeline mock runner & serialization tests
│   ├── test_utils.py            # Dataset streaming, GT extraction & metrics tests
│   └── test_cli.py              # CLI argument parsing & execution tests
│
├── configs/
│   └── default_inference.yaml   # Default slicing (640×640, 20% overlap) & GreedyNMM thresholds
│
├── models/
│   └── README.md                # Expected ONNX weight specifications & signatures
│
├── licenses/                    # Upstream and repository licenses
│   ├── LICENSE-APACHE-2.0.txt
│   ├── LICENSE-RT-DETRv4.txt    # RT-DETR Apache-2.0 License
│   ├── LICENSE-AGAR-DATASET.txt # AGAR Dataset CC BY-NC 2.0 License
│   └── LICENSE-SAHI.txt         # SAHI MIT License
├── scripts/                     # Developer utility scripts (dependency upgrades)
│   └── update_dependencies.sh
├── Makefile                     # Developer automation (install, test, lint, typecheck, hooks)
├── pyproject.toml               # Package build specifications (uv_build, ruff, ty)
├── .pre-commit-config.yaml      # Git hygiene hooks
├── .gitignore                   # Excludes weights, checkpoints, caches, and datasets
├── .gitattributes               # LF line ending normalization & binary definitions
├── AGENTS.md                    # Developer standards and agent guidelines
├── LICENSE                      # Repository license (Apache-2.0)
├── NOTICE                       # Third-party attribution & CC BY-NC 2.0 notices
└── README.md                    # This document
```

---

## Quickstart

### 1. Developer Setup
Initialize the virtual environment, install the inference engine in editable mode with development dependencies, and register git pre-commit hooks:

```bash
make install
```

*(Note: Automatically uses `uv` if present on your system, falling back to standard `python3 -m venv` and `pip`.)*

### 2. Run Tests, Lints & Type Checks
```bash
# Run unit tests
make test

# Run linter and static type checker (ruff & ty)
make lint

# Run static type checker (ty)
make typecheck

# Format Python code (ruff)
make format

# Build Sphinx HTML documentation
make docs

# Clean Sphinx build artifacts
make docs-clean

# Simulate GitHub Actions CI locally
make act

# Upgrade dependencies and run verification checks
make update-deps
```

---

## Inference Engine Usage

### Command-Line Interface (`autocfu`)

```bash
# Sliced inference on a single Petri dish with visual overlay
autocfu --image path/to/plate.jpg \
        --model models/rtdetrv4_r18vd_640.onnx \
        --output-dir results/ \
        --save-overlay

# Batch process an entire directory and export summary CSV
autocfu --input-dir path/to/plates/ \
        --model models/rtdetrv4_r18vd_640.onnx \
        --output-dir results/ \
        --save-csv \
        --save-json
```

*(Note: `autocfu-infer` and `agar-infer` are also available as command aliases).*

### CLI Options
- `--image`: Path to single plate image file.
- `--input-dir`: Directory containing images to process.
- `--model`: Path to exported RT-DETRv4 ONNX model.
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

### Python API

```python
from autocfu import AutoCFUEngine, load_config

# Load config (defaults to 640x640 slices, 20% overlap, GreedyNMM match threshold 0.50)
config = load_config("configs/default_inference.yaml")

# Initialize engine with your trained ONNX model
engine = AutoCFUEngine(model_path="models/rtdetrv4_r18vd_640.onnx", config=config)

# Run full-plate inference
result = engine.infer_image("plate.jpg")

print(f"Colonies detected: {result.colony_count}")
print(f"Grid processed: {result.num_slices} tiles in {result.execution_time_seconds:.2f}s")

# Save visual overlay
result.draw_overlay(output_path="results/plate_annotated.jpg")
```

---

## Model Deployment & ONNX Export

The `autoCFU` inference engine operates independently from model training pipelines:
- **Architecture**: Optimized for RT-DETRv4 colony detection models with Slicing-Aided Hyper Inference (SAHI).
- **ONNX Compatibility**: Accepts standard exported ONNX models conforming to the input and output tensor contracts detailed in [models/README.md](models/README.md).
- **Lightweight Runtime**: Runs purely on ONNX Runtime without requiring PyTorch, CUDA toolchains, or heavy training dependencies during deployment.

---

## Licensing & Compliance

- **Software Code**: Licensed under the [Apache License, Version 2.0](LICENSE).
- **AGAR Dataset & Derived Model Weights**:
  The AGAR dataset is provided under **Creative Commons Attribution-NonCommercial 2.0 Generic (CC BY-NC 2.0)**.
  - Any model weights fine-tuned on AGAR (including weights placed in `models/`) are restricted to **academic, educational, and non-commercial research use only**.
  - See [NOTICE](NOTICE) and [licenses/LICENSE-AGAR-DATASET.txt](licenses/LICENSE-AGAR-DATASET.txt) for complete details.
- **Upstream Attributions**:
  - **RT-DETR / RT-DETRv4**: Apache-2.0 License, Copyright (c) 2025 The RT-DETRv4 Authors.
  - **SAHI**: MIT License, Copyright (c) 2021 Fatih Cagatay Akyon.
