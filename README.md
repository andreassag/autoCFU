# AutoCFU — Standalone Multi-Model Petri Dish Inference Engine

[![CI](https://github.com/andreassag/autoCFU/actions/workflows/ci.yml/badge.svg)](https://github.com/andreassag/autoCFU/actions/workflows/ci.yml)
[![Docs](https://github.com/andreassag/autoCFU/actions/workflows/docs.yml/badge.svg)](https://github.com/andreassag/autoCFU/actions/workflows/docs.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**AutoCFU** is a lightweight, dependency-minimal inference engine implemented in both **Python** and **Rust** for automated Petri dish analysis using exported ONNX Runtime models. It supports three commercially permissive model architectures with automatic model-type detection:

1. **Radial CenterNet (`centernet`)** — Keypoint center localization and colony radius estimation `(cx, cy, r_px)`.
2. **2D StarDist (`stardist`)** — 32-ray star-convex polygon instance segmentation with `uint16` instance mask rasterization (`.tif` / `.png`).
3. **Plate & Section Countability Classifier (`countability`)** — 3-class plate/section classification (`empty`, `countable`, `uncountable`) with binary countability verdicts and grid overlays.

> All source code, slicing/merging algorithms, and supported ONNX architectures in `autoCFU` use commercially permissive open-source licenses (**Apache-2.0**, **MIT**, **BSD-3-Clause**). See [NOTICE](NOTICE) for third-party attributions and dataset proof-of-concept notes.

---

## Key Features

- **Zero Heavy ML Frameworks**: Requires only ONNX Runtime (`onnxruntime` in Python, `ort` in Rust) — no PyTorch, CUDA toolkit, or training dependencies.
- **High-Resolution Slicing-Aided Inference (SAHI)**: Processes 4000×4000 agar plates via overlapping 640×640 tiles with seam-clamped boundary coverage.
- **Scale-Invariant Cross-Tile Merging**: `GreedyNMM` and `NMS` support `center_dist` (`d / min(r1, r2)`), `ios`, and `iou`, performing confidence-weighted averaging of colony centers, radii, and 32-ray StarDist polygons across tile overlaps.
- **Auto-Detected ONNX Signatures**: Automatically identifies `centernet`, `stardist`, or `countability` graphs from output tensor names (or accepts `--model-type`).
- **Dual Language Mathematical Parity ($\le 10^{-6}$)**: Identical algorithms, configuration schemas, and CLI flags across Python `autocfu` and native Rust `autocfu`.

---

## Quick Start

### Python Installation & Usage

```bash
uv pip install -e .
```

Run standalone ONNX inference via CLI:

```bash
# Colony detection (Radial CenterNet)
autocfu --image plate.jpg --model models/centernet.onnx \
  --save-overlay --save-json --save-csv --output-dir outputs/

# Instance segmentation (2D StarDist 32-ray) with uint16 instance mask export
autocfu --image plate.jpg --model models/stardist.onnx \
  --save-overlay --save-mask --save-json --output-dir outputs/

# Countability screening across whole plate + 640x640 sections
autocfu --image plate.jpg --model models/countability.onnx \
  --countability-scope both --uncountable-threshold 0.50 \
  --save-overlay --save-json --save-csv --output-dir outputs/
```

Python API:

```python
from autocfu import AutoCFUEngine, EngineConfig, load_config

config = load_config("configs/default_inference.yaml")
engine = AutoCFUEngine("models/stardist.onnx", config=config)

result = engine.predict("plate.jpg")
print(f"Detected {result.colony_count} colonies (radii mean: {sum(result.radii)/max(1, len(result.radii)):.1f} px)")
result.save_mask("outputs/plate_mask.tif")
result.draw_overlay(output_path="outputs/plate_overlay.png")
```

### Rust Installation & Usage

Build the native Rust CLI and library:

```bash
cargo build --release
```

Run native ONNX inference with identical CLI flags:

```bash
./target/release/autocfu \
  --image plate.jpg \
  --model models/stardist.onnx \
  --save-overlay --save-mask --save-json --save-csv \
  --output-dir outputs/
```

Rust library API:

```rust
use autocfu::{AutoCFUEngine, EngineConfig};

let cfg = EngineConfig::default();
let mut engine = AutoCFUEngine::new("models/centernet.onnx", cfg)?;
let result = engine.predict_file("plate.jpg")?;
println!("Colonies detected: {}", result.colony_count);
```

---

## Documentation & Development

Run quality checks across both languages:

```bash
make format lint test docs
make rust-fmt rust-lint rust-test rust-doc
```

See the full Sphinx documentation under `docs/` (`make docs`) and architecture rules in [AGENTS.md](AGENTS.md).
