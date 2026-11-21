# AGENTS.md — Developer and Agent Guidelines for AutoCFU

This document serves as the canonical technical guide and operating standard for AI coding agents and human contributors working within the `autoCFU` repository. All contributions must adhere to the architectural boundaries, coding standards, verification procedures, and checklists detailed below.

---

## 1. System Architecture & Repository Organization

### 1.1 Decoupled Architecture
`autoCFU` is a production-grade, lightweight inference engine and command-line utility implemented in **both Python and Rust** for automated detection, star-convex instance segmentation, and countability classification of microbial colonies on Petri dishes.

The system is architecturally decoupled from upstream model creation:
- **`autoCFU` (This Repository)**: Hosts the standalone Python (`src/autocfu/`) and Rust (`rust/src/`) inference engines, tiling slicer, spatial merger, 16-bit mask rasterizer, and ONNX Runtime execution pipelines. It intentionally excludes heavyweight machine learning frameworks (such as PyTorch, TorchVision, or CUDA compiler toolchains) from runtime dependencies to keep deployment lightweight and portable.
- **Standalone ONNX Artifact Contract**: `autoCFU` depends strictly on exported `.onnx` model weights conforming to the documented tensor contracts in Section 1.3 and auto-detects the model family (`centernet`, `stardist`, or `countability`) at load time.

```
autoCFU/
├── .github/
│   ├── dependabot.yml              # Weekly dependency updates (pip, cargo, github-actions)
│   └── workflows/
│       ├── ci.yml                  # Python (3.10-3.12) & Rust (fmt, clippy, test, doc) CI
│       ├── dependabot-auto-merge.yml # Automated Dependabot PR approval and auto-merge
│       └── docs.yml                # Automated GitHub Pages Sphinx documentation deployment
├── Cargo.toml                      # Rust crate configuration (ort, ndarray, image, tiff, clap)
├── rust/
│   └── src/                        # Standalone Rust ONNX inference engine & CLI binary
│       ├── lib.rs                  # Public crate exports and parity unit tests
│       ├── main.rs                 # Native `autocfu` Rust CLI binary
│       ├── error.rs                # `AutoCFUError` thiserror enum (zero unwrap in lib paths)
│       ├── config.rs               # YAML config parser & parameter validation
│       ├── slicer.rs               # 2D overlapping grid window tiling
│       ├── merger.rs               # GreedyNMM & NMS (center_dist / ios / iou) + polygon reconstruction
│       ├── model.rs                # ONNX Runtime runners (CenterNet, StarDist, Countability)
│       └── pipeline.rs             # Sliced inference, uint16 mask rasterizer, overlays, CSV/JSON
├── configs/
│   └── default_inference.yaml      # Default slicing, merging, radial, and countability settings
├── docs/                           # Sphinx documentation suite (Furo theme, Python & Rust guides)
├── src/
│   └── autocfu/                    # Standalone Python inference engine (`uv_build`)
│       ├── __init__.py             # Public exports
│       ├── cli.py                  # CLI entry point (autocfu, autocfu-infer, agar-infer)
│       ├── config.py               # Configuration schema and YAML parser
│       ├── merger.py               # GreedyNMM & NMS (center_dist / ios / iou)
│       ├── model.py                # ONNX Runtime runners & auto-detector
│       ├── pipeline.py             # Sliced inference orchestrator, overlays, uint16 mask export
│       ├── slicer.py               # 2D grid window tiling and coordinate translation
│       └── utils.py                # Inspection, ground-truth extraction, metrics, and visualization
├── tests/                          # Python unit tests & Python/Rust mathematical parity tests
├── models/
│   └── README.md                   # ONNX model artifact specifications across all 3 families
├── licenses/                       # Permissive software licenses (Apache-2.0, MIT, BSD-3-Clause) & data notice
├── NOTICE                          # Commercial software guarantee & proof-of-concept dataset notice
├── pyproject.toml                  # Python package configuration, ruff, and ty
└── AGENTS.md                       # This guideline document
```

### 1.2 Slicing-Aided Inference & Countability Evaluation Pipeline
High-resolution Petri dish photography typically captures agar plates at 4000×4000 pixels. Directly downscaling a 4000×4000 image to standard 640×640 detector inputs reduces minute microbial colonies (<15 pixels in diameter) to sub-pixel noise.

For **spatial colony models (`centernet` and `stardist`)**, `autoCFU` implements a Slicing-Aided Hyper Inference (SAHI) pipeline in both Python and Rust:
1. **Grid Generation (`slicer`)**: Slices the input dish into overlapping tiles (default: 640×640, 20% overlap). Step stride is calculated as:
   $$\text{step} = \lfloor \text{slice\_size} \times (1.0 - \text{overlap}) \rfloor$$
   Boundary edges are anchored/clamped so right and bottom borders remain fully covered without zero padding.
2. **ONNX Tile Execution (`model`)**: Runs each slice through the loaded CenterNet or 2D StarDist ONNX model (supporting both baked-in decoded tensor outputs and raw-map decoding fallback).
3. **Global Coordinate Mapping (`pipeline`)**: Translates bounding boxes `(x1, y1, x2, y2)`, centers `(cx, cy)`, and StarDist 32-vertex polygons `(px_k, py_k)` back into global dish coordinates.
4. **Greedy Non-Maximum Merging (`merger`)**: Deduplicates colonies straddling tile seams using `GreedyNMM` or `NMS` under `center_dist` (`d / min(r_1, r_2)`), `ios`, or `iou`. `GreedyNMM` averages centers, radii, bounding boxes, and StarDist radial vectors `rays[N]` weighted by confidence, reconstructing polygons via:
   $$x_k = c_x + r_k \cos\!\left(\frac{2\pi k}{N}\right), \quad y_k = c_y + r_k \sin\!\left(\frac{2\pi k}{N}\right)$$
5. **16-bit Instance Mask Rasterization (`pipeline`)**: Rasterizes StarDist polygons or CenterNet circles into 16-bit `uint16` instance label masks (`.tif` / `.png`) with unique colony IDs `1..N`.

For **Countability classifier models (`countability`)**, `autoCFU` evaluates the plate in `--countability-scope` mode (`both`, `plate`, or `sections`), running ONNX classification over the resized whole plate and/or each 640×640 section tile to emit 3-class probabilities `(empty, countable, uncountable)` and binary countability verdict:
$$\text{is\_countable} = \bigl(P(\text{uncountable}) < \tau_{\text{uncountable}}\bigr)$$

### 1.3 ONNX Tensor Contracts (Auto-Detected)
`autoCFU` automatically inspects ONNX output tensor names (or custom metadata key `autocfu_model_type`) to select the appropriate runner (`centernet`, `stardist`, or `countability`), unless overridden with `--model-type`.

#### Shared Input Contract
| Input Name | Shape | Dtype | Range / Layout | Description |
| :--- | :--- | :--- | :--- | :--- |
| `images` | `[batch_size, 3, H, W]` | `float32` | `[0.0, 1.0]` (RGB, NCHW) | Normalized input image batch (default `640×640`). |

#### Model 1 — Radial CenterNet (`centernet`) Outputs
| Tensor Name | Shape | Dtype | Content |
| :--- | :--- | :--- | :--- |
| `boxes` | `[B, K, 4]` | `float32` | Bounding boxes `[xmin, ymin, xmax, ymax]` in tile pixels. |
| `scores` | `[B, K]` | `float32` | Colony confidence scores in `[0.0, 1.0]`. |
| `labels` | `[B, K]` | `int64` | Foreground class IDs (`0` = `colony`). |
| `centers` | `[B, K, 2]` | `float32` | Sub-pixel colony centers `[cx, cy]`. |
| `radii` | `[B, K]` | `float32` | Predicted colony radii `r_px` in pixels. |

*(Raw-map fallback also supports dense heads `heatmap [B, C, H/4, W/4]`, `offset [B, 2, H/4, W/4]`, `radius [B, 1, H/4, W/4]`.)*

#### Model 2 — 2D StarDist (`stardist`) Outputs
| Tensor Name | Shape | Dtype | Content |
| :--- | :--- | :--- | :--- |
| `boxes` | `[B, K, 4]` | `float32` | Tight polygon bounding boxes `[xmin, ymin, xmax, ymax]`. |
| `scores` | `[B, K]` | `float32` | Object probability scores in `[0.0, 1.0]`. |
| `labels` | `[B, K]` | `int64` | Foreground class IDs (`0` = `colony`). |
| `centers` | `[B, K, 2]` | `float32` | Colony center coordinates `[cx, cy]`. |
| `rays` | `[B, K, N_rays]` | `float32` | Radial distances along `N_rays` equiangular rays (default `32`). |
| `polygons` | `[B, K, N_rays, 2]` | `float32` | Decoded polygon vertices `[x_k, y_k]` in tile pixels. |

*(Raw-map fallback also supports dense heads `prob [B, 1, H/4, W/4]` and `rays [B, N_rays, H/4, W/4]`.)*

#### Model 3 — Plate & Section Countability (`countability`) Outputs
| Tensor Name | Shape | Dtype | Content |
| :--- | :--- | :--- | :--- |
| `logits` | `[B, 3]` | `float32` | Unnormalized logits for `[empty, countable, uncountable]`. |
| `probs` | `[B, 3]` | `float32` | Softmax class probabilities `[P(empty), P(countable), P(uncountable)]`. |
| `pred_class` | `[B]` | `int64` | Argmax class ID (`0`: `empty`, `1`: `countable`, `2`: `uncountable`). |
| `is_countable` | `[B]` | `bool` | Binary verdict (`P(uncountable) < 0.50` default; re-evaluated dynamically if threshold differs). |

---

## 2. Python Coding Standards

All Python code in `autoCFU` must meet high engineering standards:

### 2.1 Modern Python 3.10+ Typing
- Target Python `>=3.10`.
- Always include `from __future__ import annotations` at the top of every Python file.
- Use built-in generic types (`list[...]`, `dict[...]`, `tuple[...]`, `set[...]`) and union operator `|`.
- Annotate all function signatures, method arguments, return types, and dataclass fields completely.

### 2.2 Google-Style Docstrings & Rationale Comments
- All public modules, classes, methods, and standalone functions must include Google-style docstrings.
- Comment non-obvious mathematical invariants, boundary clamping, and numerical guards.

### 2.3 Formatting, Linting, and Type Checking (Ruff & ty)
- Build backend: **`uv_build`** in `pyproject.toml`.
- Line length limit: **120 characters**, target **py310**, rules `E, F, W, I`.
- Static type checking is enforced with Astral's **`ty`** (`ty check src/ tests/`).
- Run `make format`, `make lint`, and `make typecheck` before committing.

---

## 3. Rust Coding Standards

All Rust code in `Cargo.toml` and `rust/src/` must follow strict safety, documentation, and parity standards:

### 3.1 Edition & Documentation
- Target **Rust 2021 edition**.
- Enable `#![warn(missing_docs)]` and provide `///` rustdoc comments on all public items, including `# Errors` sections on every fallible function.

### 3.2 Error Handling
- Use `thiserror` (`AutoCFUError`) implementing `std::error::Error`.
- Functions that may fail must return `autocfu::Result<T>`.
- **Do not use `unwrap()` or `expect()`** anywhere in library or binary production paths (`rust/src/`); reserve them strictly for `#[cfg(test)]` unit test modules.

### 3.3 Safety & FFI Conventions
- Enforce `#![deny(unsafe_op_in_unsafe_fn)]` across the crate.
- Prefer safe Rust abstractions (`ndarray`, `ort::value::TensorRef`, `image`, `tiff`). Any unavoidable `unsafe` block must include a detailed `// SAFETY:` justification.

### 3.4 Mathematical Parity
Python and Rust implementations of slicing, overlap metrics, `GreedyNMM`, `NMS`, polygon vertex reconstruction, and ONNX decoding must maintain strict numerical parity ($\le 10^{-6}$ tolerance on coordinates, centers, radii, and ray arrays).

### 3.5 Commercial Usability & Technology Exclusions
- **Only Apache-2.0 / MIT / BSD-3-Clause libraries and architectures** may be integrated into `autoCFU`. Do not add copyleft-only runtime dependencies or non-commercial model components.
- **Jupyter Notebooks (`.ipynb`), training code, and C++ are strictly excluded** from `autoCFU`.

---

## 4. Development Tooling & Makefile Targets

| Command | Description |
| :--- | :--- |
| `make install` | Create `.venv`, install `.[dev,docs]` in editable mode, configure pre-commit hooks. |
| `make test` | Execute Python test suite via `pytest tests/`. |
| `make lint` | Run `ruff check src/ tests/` and `ty check src/ tests/`. |
| `make typecheck` | Run static type checking via `ty check src/ tests/`. |
| `make format` | Format Python codebase with `ruff format src/ tests/`. |
| `make rust-fmt` | Verify/apply `cargo fmt --all`. |
| `make rust-lint` | Run `cargo clippy --all-targets -- -D warnings`. |
| `make rust-test` | Run `cargo test --all-targets`. |
| `make rust-build` | Build optimized Rust binary via `cargo build --release`. |
| `make rust-doc` | Generate Rust API documentation via `cargo doc --no-deps`. |
| `make docs` | Build Sphinx HTML documentation (`sphinx-build -b html -W docs docs/_build/html`). |
| `make pre-commit` | Run pre-commit hooks across repository files. |

---

## 5. Continuous Integration, Dependabot & Docs Deployment

- **CI (`.github/workflows/ci.yml`)**: Runs on `push` and `pull_request` to `main` and `develop`. Executes Python verification across **Python 3.10, 3.11, and 3.12** (`ruff format --check`, `ruff check`, `ty check`, `pytest`, `sphinx-build -W`) and a dedicated `rust` job (`cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test --all-targets`, `cargo doc --no-deps`).
- **Docs (`.github/workflows/docs.yml`)**: Validates Sphinx builds on PRs targeting `main` and `develop` and deploys to GitHub Pages on `main`.
- **Dependabot (`.github/dependabot.yml` + `.github/workflows/dependabot-auto-merge.yml`)**: Tracks weekly updates for `pip`, `cargo`, and `github-actions`, auto-merging only after both Python `test` and `rust` jobs succeed.

---

## 6. Change Verification Checklist

1. **Dual Implementation Parity**: Keep Python (`src/autocfu/`) and Rust (`rust/src/`) mathematically aligned ($\le 10^{-6}$) and commercial-license clean.
2. **No Training References**: Do not add training loops, dataset builders, PyTorch dependencies, or `.ipynb` files to `autoCFU`.
3. **Tests**: Update Python tests (`tests/`), parity tests (`tests/test_rust_parity.py`), and Rust unit tests (`rust/src/lib.rs`).
4. **Docs & Configs**: Update `configs/default_inference.yaml`, `README.md`, `models/README.md`, `NOTICE`, `licenses/`, and `docs/`.
5. **Checks Passing**: Verify `make format`, `make lint`, `make test`, `make docs`, `make rust-fmt`, `make rust-lint`, `make rust-test`, and `make rust-doc` exit with zero errors/warnings.
