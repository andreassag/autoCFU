# AGENTS.md — Developer and Agent Guidelines for AutoCFU

This document serves as the canonical technical guide and operating standard for AI coding agents and human contributors working within the `autoCFU` repository. All contributions must adhere to the architectural boundaries, coding standards, verification procedures, and checklists detailed below.

---

## 1. System Architecture & Repository Organization

### 1.1 Decoupled Architecture
`autoCFU` is a production-grade, lightweight inference engine and command-line utility for automated detection, segmentation, and enumeration of microbial colonies on Petri dishes.

The system is architecturally decoupled from model training and dataset preparation:
- **`autoCFU` (This Repository)**: Hosts the standalone inference engine, CLI, slicing, merging, and ONNX Runtime execution pipeline. It intentionally excludes heavyweight machine learning training frameworks (such as PyTorch, TorchVision, or CUDA build toolchains) from runtime dependencies to keep installation lightweight and portable.
- **Training & Dataset Pipelines**: Handled separately outside of this repository to maintain a minimal, dependency-light inference footprint. Models are trained, validated, and exported to ONNX format before deployment into `autoCFU`.

```
autoCFU/
├── .github/
│   ├── dependabot.yml          # Automated weekly dependency updates (pip & GitHub Actions)
│   └── workflows/
│       ├── ci.yml              # Multi-version CI matrix (Python 3.10, 3.11, 3.12)
│       ├── dependabot-auto-merge.yml # Automated Dependabot PR approval and auto-merge
│       └── docs.yml            # Automated GitHub Pages Sphinx documentation deployment
├── docs/                       # Sphinx documentation suite (Furo theme, Guides, API reference)
├── configs/
│   └── default_inference.yaml  # Default slicing, overlap, and merger parameters
├── src/
│   └── autocfu/                # Standalone Python inference engine
│       ├── __init__.py         # Public exports
│       ├── cli.py              # CLI entry point (autocfu, autocfu-infer, agar-infer)
│       ├── config.py           # Configuration schema and YAML parser
│       ├── merger.py           # Greedy Non-Maximum Merging (GreedyNMM & NMS)
│       ├── model.py            # ONNX Runtime session manager and post-processing
│       ├── pipeline.py         # Full-dish sliced inference orchestrator
│       ├── slicer.py           # 2D grid window tiling and coordinate translation
│       └── utils.py            # Inspection, dataset streaming, and visualization helpers
├── tests/                      # Unit and integration test suite
├── models/
│   └── README.md               # Model artifact specifications and export instructions
├── scripts/
│   └── update_dependencies.sh  # Automated dependency update and verification script
├── pyproject.toml              # Package configuration, dependencies, uv_build, ruff, and ty
├── Makefile                    # Standard developer automation targets
├── .pre-commit-config.yaml     # Pre-commit hooks for code hygiene
└── AGENTS.md                   # This guideline document
```

### 1.2 Slicing-Aided Sliced Inference Pipeline
High-resolution Petri dish photography typically captures agar plates at 4000×4000 pixels. Directly downscaling a 4000×4000 image to standard detector input sizes (e.g., 640×640) reduces minute microbial colonies (<15 pixels in diameter) to unrecognizable sub-pixel noise.

`autoCFU` implements a Slicing-Aided Hyper Inference (SAHI) pipeline:
1. **Grid Generation (`slicer.py`)**: Slices the input dish into overlapping tiles of fixed dimensions (default: 640×640) with configured overlap (default: 20%). Step stride is calculated as:
   $$\text{step} = \lfloor \text{slice\_size} \times (1.0 - \text{overlap}) \rfloor$$
   Boundary edges are clamped to ensure full image coverage without empty padding artifacts.
2. **Batch ONNX Inference (`model.py`)**: Runs each image slice through the exported RT-DETR ONNX model.
3. **Global Coordinate Mapping (`pipeline.py`)**: Translates bounding box coordinates from slice-local coordinates $(x_1, y_1, x_2, y2)$ back to global dish coordinates $(x_1 + x_{\text{offset}}, y_1 + y_{\text{offset}}, x_2 + x_{\text{offset}}, y_2 + y_{\text{offset}})$.
4. **Greedy Non-Maximum Merging (`merger.py`)**: Merges redundant detections across overlapping tile boundaries using GreedyNMM or standard NMS, clustering intersecting bounding boxes and computing confidence-weighted centroid coordinates:
   $$\mathbf{box}_{\text{merged}} = \frac{\sum_{i} \text{score}_i \times \mathbf{box}_i}{\sum_{i} \text{score}_i}$$

### 1.3 Upstream ONNX Tensor Contract (RT-DETRv4)
Models are exported using the RT-DETRv4 ONNX export pipeline. The `autoCFU` model runner (`RTDetrV4ONNXModel` / `RTDetrONNXModel`) expects the following tensor contracts:

#### Inputs
| Input Name | Shape | Dtype | Range / Layout | Description |
| :--- | :--- | :--- | :--- | :--- |
| `images` | `[batch_size, 3, 640, 640]` | `float32` | `[0.0, 1.0]` (RGB, NCHW) | Normalized input image batch. |
| `orig_target_sizes` | `[batch_size, 2]` | `int64` | `[[height, width]]` | Original slice dimensions (`[[640, 640]]`), used by RT-DETR coordinate post-processors. |

#### Outputs
| Output Candidates | Shape | Dtype | Content |
| :--- | :--- | :--- | :--- |
| `labels`, `pred_labels`, `classes` | `[batch_size, num_queries]` | `int64` | Predicted class IDs (`0` corresponds to `colony`). |
| `boxes`, `pred_boxes`, `bboxes` | `[batch_size, num_queries, 4]` | `float32` | Pixel coordinates in `[x1, y1, x2, y2]` format. |
| `scores`, `pred_scores`, `confs` | `[batch_size, num_queries]` | `float32` | Prediction confidence scores in `[0.0, 1.0]`. |

*Note on Architecture Parity*:
- `_find_output_tensor` searches across candidate output names for compatibility across RT-DETRv4 and custom ONNX graph exporters.
- `_safe_unbatch` safely strips leading batch dimensions when `batch_size == 1`.
- If `orig_target_sizes` is not accepted by the ONNX graph and resizing occurred, coordinates are dynamically rescaled by $(w_{\text{orig}} / w_{\text{input}}, h_{\text{orig}} / h_{\text{input}})$.

---

## 2. Python Coding Standards

All Python code in `autoCFU` must meet high engineering standards:

### 2.1 Modern Python 3.10+ Typing
- Target Python `>=3.10`.
- Always include `from __future__ import annotations` at the top of every Python file.
- Use built-in generic types for collections: `list[...]`, `dict[...]`, `tuple[...]`, `set[...]` (do not import `List`, `Dict`, `Tuple`, `Set` from `typing`).
- Use the union operator `|` for optional types and type unions: `str | None` or `int | float` (prefer over `Optional[...]` or `Union[...]` in new code).
- Annotate all function signatures, method arguments, return types, and dataclass fields completely.

```python
# GOOD:
def calculate_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ...

# AVOID:
from typing import Tuple
def calculate_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ...
```

### 2.2 Google-Style Docstrings
All public modules, classes, methods, and standalone functions must include Google-style docstrings:

```python
def merge_detections(
    detections: list[Detection],
    match_threshold: float = 0.5,
    match_metric: str = "iou",
) -> list[Detection]:
    """Merge overlapping detections using Greedy Non-Maximum Merging.

    Iterates over detections in descending order of confidence and clusters
    intersecting detections exceeding the match metric threshold, calculating
    a confidence-weighted average bounding box for each cluster.

    Args:
        detections: List of raw detection bounding boxes with scores.
        match_threshold: Metric threshold for clustering (e.g., IoU >= 0.5).
        match_metric: Overlap calculation metric ('iou' or 'ios').

    Returns:
        List of merged Detection objects with updated bounding boxes and scores.

    Raises:
        ValueError: If match_metric is not supported.
    """
```

### 2.3 Comments & Mathematical Rationale
- Explain the **"why"**, not just the "what".
- Document mathematical coordinate transformations, clipping edge cases, and algorithm choices (e.g., why IoS is advantageous for detecting clustered small colonies adjacent to larger ones).
- Keep module header docstrings clear and concise.

### 2.4 Formatting, Linting, and Type Checking (Ruff & ty)
- Build backend: **`uv_build`** in `pyproject.toml`.
- Line length limit: **120 characters**.
- Target Python version: **py310** (`python-version = "3.10"`).
- Enabled lint rules: `E` (pycodestyle errors), `F` (Pyflakes), `W` (pycodestyle warnings), `I` (isort import sorting).
- Static type checking is enforced with Astral's **`ty`** (`ty check src/ tests/`).
- Run `make format`, `make lint`, and `make typecheck` before committing. Code must pass with zero errors and zero warnings.

---

## 3. Rust Coding Standards

When adding or modifying native Rust acceleration modules (e.g., high-performance geometric tiling or NMM spatial indexing in Rust):

### 3.1 Edition & Documentation
- Target **Rust 2021 edition** (or later).
- Every public struct, enum, trait, function, and method must have complete `///` rustdoc documentation.
- Include `# Errors` and `# Panics` sections in rustdoc where appropriate.

### 3.2 Error Handling
- Use idiomatic error handling via `thiserror` (for library crates) or custom enum errors implementing `std::error::Error`.
- Functions that may fail must return `Result<T, E>`.
- **Do not use `unwrap()` or `expect()`** in library or production paths. Handle errors explicitly or propagate with `?`.
- Never trigger panics in library code.

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SlicingError {
    #[error("Invalid tile dimensions: {0}x{1}")]
    InvalidDimensions(u32, u32),
    #[error("Overlap factor must be in [0.0, 1.0), got {0}")]
    InvalidOverlap(f32),
}
```

### 3.3 Safety & FFI Conventions
- Isolate any native Foreign Function Interface (FFI) or PyO3 bindings behind safe, idiomatic Rust wrapper APIs.
- Enforce `#![deny(unsafe_op_in_unsafe_fn)]` across the crate.
- Every `unsafe` block must be strictly minimal and preceded by an explicit `// SAFETY:` comment documenting the invariants, pointer validity, and alignment guarantees.

```rust
// SAFETY:
// 1. `ptr` is non-null and properly aligned for `f32`.
// 2. The slice length `len` corresponds exactly to the allocated memory buffer.
// 3. No mutable alias exists for the lifetime of this reference.
let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
```

### 3.4 Mathematical Parity
Any Rust implementation of algorithms existing in Python (such as `slicer` or `merger`) must maintain strict mathematical parity with Python:
- **Tiling coordinates**: Stride calculation `step = ((size as f32) * (1.0 - overlap)).floor() as u32`. Edge tiles must clip to image boundaries identical to Python's `min(x + size, img_w)`.
- **IoU / IoS**: Bounding box coordinate calculations $(x_1, y_1, x_2, y_2)$ must use identical intersection and union formulas with floating-point tolerance $\le 10^{-6}$.
- **Centroid Merging**: Weighted coordinate averages must match `sum(score_i * coord_i) / sum(score_i)`.

### 3.5 Technology Exclusions
- **Jupyter Notebooks (`.ipynb`) are strictly excluded** from `autoCFU`. Training notebooks and interactive experiments belong outside this inference repository.
- **C++ is strictly excluded**. Rust is the designated native language for systems-level performance extensions.

---

## 4. Development Tooling & Makefile Targets

The repository includes a comprehensive `Makefile` providing standardized developer automation:

| Command | Description |
| :--- | :--- |
| `make install` | Creates virtual environment (`uv` preferred, fallback to `python3 -m venv`), installs `.[dev,docs]` in editable mode, and configures pre-commit hooks. |
| `make test` | Executes unit tests via `pytest tests`. |
| `make lint` | Runs `ruff check src/ tests/` and `ty check src/ tests/` across the codebase. |
| `make typecheck` | Runs static type checking via `ty check src/ tests/`. |
| `make format` | Formats Python code using `ruff format src/ tests/`. |
| `make docs` | Builds Sphinx HTML documentation (`sphinx-build -b html -W docs docs/_build/html`). |
| `make docs-clean` | Removes Sphinx documentation build artifacts (`rm -rf docs/_build`). |
| `make pre-commit` | Runs pre-commit hooks across all staged and repository files. |
| `make clean` | Cleans build artifacts, egg metadata, and Python test caches. |
| `make act` | Executes GitHub Actions CI workflow locally using `act` / `gh act`. |
| `make update-deps` | Executes `scripts/update_dependencies.sh` to upgrade dependencies and run formatting, linting, type checking, and tests. |

---

## 5. Continuous Integration, Dependabot & Local Verification

### 5.1 GitHub Actions CI (`.github/workflows/ci.yml`)
- Automatically runs on `push` and `pull_request` to the `main` branch.
- **Trigger Guardrails (`paths-ignore`)**: To avoid redundant test matrix runs, CI is skipped when commits/PRs modify only documentation (`docs/**`, `notebooks/**`, `**/*.md`), legal assets (`LICENSE*`, `NOTICE*`, `licenses/**`), git configurations (`.gitignore`, `.gitattributes`), pre-commit hooks (`.pre-commit-config.yaml`), or model card metadata (`models/**`). Any change touching Python package source code (`src/`), tests (`tests/`), dependencies (`pyproject.toml`), or CI workflows triggers the full pipeline.
- Test matrix covers **Python 3.10, 3.11, and 3.12**.
- CI workflow execution steps:
  1. Check out repository via `actions/checkout@v7`.
  2. Set up Python runtime via `actions/setup-python@v7` and `astral-sh/setup-uv@v7`.
  3. Install dependencies: `uv pip install --system -e ".[dev,docs]"`.
  4. Format verification: `ruff format --check src/ tests/`.
  5. Linter verification: `ruff check src/ tests/`.
  6. Static type checking: `ty check src/ tests/`.
  7. Test execution: `pytest tests/`.
  8. Sphinx doc build: `sphinx-build -b html -W docs docs/_build/html`.

### 5.2 Dependabot Configuration (`.github/dependabot.yml`)
- Configured with `version: 2`.
- Automatically checks for updates weekly:
  - `pip` ecosystem: monitors `/` (`pyproject.toml`).
  - `github-actions` ecosystem: monitors `/` (`.github/workflows/ci.yml`).
- Caps open pull requests at 10 to prevent backlog noise.

### 5.3 Dependabot Auto-Merge Workflow
Dependabot pull requests are automatically approved and merged once all CI checks succeed:
- **Approval & Auto-Merge Queueing** (`.github/workflows/dependabot-auto-merge.yml`): Runs on Dependabot pull request events (`opened`, `synchronize`, `reopened`). Evaluates metadata using `dependabot/fetch-metadata@v3`, approves the pull request, and queues auto-merge with `gh pr merge --auto`.
- **Post-CI Verification Gate** (`.github/workflows/ci.yml`): The `dependabot-automerge` job depends strictly on `test` matrix completion (`needs: test`). It ensures that pull requests are merged only after format checks, lint checks, type checks, and unit tests pass across all supported Python versions (3.10, 3.11, 3.12).
- **Merge Strategy**: Attempts merge commit (`--merge`) with graceful fallback to squash (`--squash`) depending on repository branch settings.
- **Required Repository Permissions**:
  1. *Allow GitHub Actions to approve PRs*: In **Settings > Actions > General > Workflow permissions**, enable **"Allow GitHub Actions to create and approve pull requests"**.
  2. *Allow auto-merge*: In **Settings > General > Pull Requests**, enable **"Allow auto-merge"**.

### 5.4 Local CI Simulation via `act`
To run CI workflows locally without pushing commits to GitHub:
```bash
make act
# or directly:
gh act pull_request
```
*(Requires `act` or the GitHub CLI extension `nektos/gh-act` with Docker running.)*

### 5.5 GitHub Pages Documentation Deployment (`.github/workflows/docs.yml`)
- Automatically builds Sphinx HTML documentation on `pull_request` (verifying PRs that touch `docs/**`, `src/**`, `pyproject.toml`, or `.github/workflows/docs.yml` without deploying) and upon `push` to `main` or via `workflow_dispatch`.
- Builds with strict warning-as-error flag (`-W`) to prevent broken cross-references or malformed docstrings.
- Generates `docs/_build/html/.nojekyll` and uploads the artifact via `actions/upload-pages-artifact@v4` on pushes to `main`.
- Deploys to GitHub Pages via `actions/deploy-pages@v4` in the `github-pages` environment upon merge/push to `main`.
- *Repository Setting*: In GitHub repository settings under **Settings > Pages > Build and deployment**, ensure **Source** is set to **GitHub Actions**.

---

## 6. What to Update When Making Changes (Checklist)

Whenever modifying existing features or introducing new capabilities, follow this step-by-step checklist:

1. **Code Implementation (`src/autocfu/`)**
   - Ensure code conforms to Python 3.10+ typing standards or Rust 2021+ guidelines.
   - Do not add heavy training dependencies (e.g., PyTorch) to `pyproject.toml`.
   - Never commit `.ipynb` notebooks or C++ files to this repository.

2. **Test Coverage (`tests/`)**
   - Add or update corresponding unit tests in `tests/`.
   - Cover boundary conditions: empty images, zero detections, single-tile vs multi-tile images, invalid inputs.

3. **Documentation & Configurations**
   - If CLI options or inference defaults change, update `configs/default_inference.yaml`, `README.md`, and `docs/cli.rst`.
   - If ONNX tensor signatures change, update `models/README.md` and document the contract in `AGENTS.md` and `docs/index.rst`.
   - If new public modules or methods are added, update `docs/api/`.

4. **Formatting, Linting & Type Checking**
   - Format: `make format`
   - Lint & Type-Check: `make lint` / `make typecheck` (must pass with 0 errors in both `ruff` and `ty`)

5. **Unit Tests & Documentation Build**
   - Run tests: `make test` (all tests must pass)
   - Run docs: `make docs` (Sphinx build must pass with 0 warnings/errors)

6. **Pre-commit Checks**
   - Run: `make pre-commit`

7. **Local CI Verification**
   - Run: `make act` (if Docker environment is available) or run multi-version matrix checks.
