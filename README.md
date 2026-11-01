# AutoCFU: High-Resolution Microbial Colony Detection & Counting

An automated, high-resolution colony detection and enumeration system for microbiological Petri dishes using **RT-DETRv2** and **Slicing-Aided Hyper Inference (SAHI / GreedyNMM)**.

---

## Architecture & Repository Organization

Full-dish agar photography typically captures Petri dishes at 4000×4000 resolution. Direct downscaling reduces minute colonies (<15 pixels) to unrecognizable noise. This repository hosts the decoupled, production-grade Python inference engine and CLI. Dataset preparation, fine-tuning, and benchmarking are hosted in the companion repository [autoCFU-training](https://github.com/andreassag/autoCFU-training).

```
autocfu/
├── inference/                   # Standalone Python Inference Engine
│   ├── pyproject.toml           # Package definition (autocfu, autocfu CLI)
│   ├── README.md                # Engine documentation and API references
│   ├── autocfu/
│   │   ├── __init__.py          # Public exports
│   │   ├── config.py            # Dataclass schemas and YAML loader
│   │   ├── slicer.py            # 2D Grid Window Tiler (step = size * (1 - overlap))
│   │   ├── merger.py            # Greedy Non-Maximum Merging (GreedyNMM & NMS)
│   │   ├── model.py             # ONNX Runtime session manager (CUDA & CPU)
│   │   ├── pipeline.py          # Full-plate end-to-end sliced inference orchestrator
│   │   └── cli.py               # autocfu CLI entrypoint
│   └── tests/
│       ├── test_slicer.py       # Slicer grid offsets & boundary tests
│       ├── test_merger.py       # IoU and weighted coordinate clustering tests
│       ├── test_model.py        # ONNX session unbatching & provider tests
│       ├── test_pipeline.py     # Pipeline mock runner & serialization tests
│       └── test_cli.py          # CLI argument parsing & execution tests
│
├── configs/
│   └── default_inference.yaml   # Default slicing (640×640, 20% overlap) & GreedyNMM thresholds
│
├── models/
│   └── README.md                # Expected ONNX weight specifications & signatures
│
├── licenses/                    # Upstream and repository licenses
│   ├── LICENSE-APACHE-2.0.txt
│   ├── LICENSE-RT-DETRv2.txt    # RT-DETRv2 Apache-2.0 License
│   ├── LICENSE-AGAR-DATASET.txt # AGAR Dataset CC BY-NC 2.0 License
│   └── LICENSE-SAHI.txt         # SAHI MIT License
├── Makefile                     # Developer automation (install, test, lint, hooks)
├── .pre-commit-config.yaml      # Git hygiene hooks
├── .github/                     # GitHub Actions CI and Dependabot workflows
├── scripts/                     # Developer utility scripts (dependency upgrades)
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

### 2. Run Tests & Lints
```bash
# Run unit tests
make test

# Run linter
make lint

# Run pre-commit checks across all files
make pre-commit

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
        --model models/rtdetrv2_r18vd_agar_640.onnx \
        --output-dir results/ \
        --save-overlay

# Batch process an entire directory and export summary CSV
autocfu --input-dir path/to/plates/ \
        --model models/rtdetrv2_r18vd_agar_640.onnx \
        --output-dir results/ \
        --save-csv \
        --save-json
```

*(Note: `autocfu-infer` and `agar-infer` are also available as command aliases).*

### Python API

```python
from autocfu import AutoCFUEngine, load_config

# Load config (defaults to 640x640 slices, 20% overlap, GreedyNMM match threshold 0.50)
config = load_config("configs/default_inference.yaml")

# Initialize engine with your trained ONNX model
engine = AutoCFUEngine(
    model_path="models/rtdetrv2_r18vd_agar_640.onnx",
    config=config
)

# Run full-plate inference
result = engine.infer_image("plate.jpg")

print(f"Colonies detected: {result.colony_count}")
print(f"Grid processed: {result.num_slices} tiles in {result.execution_time_seconds:.2f}s")

# Save visual overlay
result.draw_overlay(output_path="results/plate_annotated.jpg")
```

---

## Research & Training Pipeline

Model training, dataset preparation, and evaluation pipelines are maintained in the companion repository:
- **Repository**: [autoCFU-training](https://github.com/andreassag/autoCFU-training)
- **Features**:
  - Google Colab notebooks for AGAR dataset streaming, 640×640 COCO tiling, and RT-DETRv2 fine-tuning.
  - Anticipated pure-Python training and benchmarking pipeline with PyTorch TensorBoard dashboard integration.
  - ONNX export scripts to generate models for this inference engine.


---

## Licensing & Compliance

- **Software Code**: Licensed under the [Apache License, Version 2.0](LICENSE).
- **AGAR Dataset & Derived Model Weights**:
  The AGAR dataset is provided under **Creative Commons Attribution-NonCommercial 2.0 Generic (CC BY-NC 2.0)**.
  - Any tiled dataset derivatives generated by `01-prepare-AGAR-dataset.ipynb` and any model weights fine-tuned on AGAR (including weights placed in `models/`) are restricted to **academic, educational, and non-commercial research use only**.
  - See [NOTICE](NOTICE) and [licenses/LICENSE-AGAR-DATASET.txt](licenses/LICENSE-AGAR-DATASET.txt) for complete details.
- **Upstream Attributions**:
  - **RT-DETRv2**: Apache-2.0 License, Copyright (c) 2024 lyuwenyu.
  - **SAHI**: MIT License, Copyright (c) 2021 Fatih Cagatay Akyon.
