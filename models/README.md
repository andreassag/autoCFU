# ONNX Model Contracts & Placement Guide

Place exported `.onnx` model weight files in this directory (`models/`). Binary weight files (`*.onnx`, `*.pt`, `*.pth`) are excluded from Git tracking via `.gitignore`.

`autoCFU` automatically identifies which architecture a loaded `.onnx` file belongs to by inspecting output tensor names (and optional ONNX metadata key `autocfu_model_type`), unless explicitly specified via `--model-type`.

---

## Supported ONNX Signatures (Opset 17+)

### Common Input Tensor
- **`images`**: `float32 [batch_size, 3, H, W]` — RGB image tensor normalized to `[0.0, 1.0]` (default `H = W = 640`).

---

### 1. Radial CenterNet (`centernet`)
Detects colony centers `(cx, cy)` and optical radii `r_px` using a ResNet+FPN backbone with commercially permissive Apache-2.0/MIT/BSD-3-Clause components.

**Primary Decoded Outputs:**
| Output Name | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `boxes` | `[B, K, 4]` | `float32` | Square enclosing boxes `[cx - r, cy - r, cx + r, cy + r]`. |
| `scores` | `[B, K]` | `float32` | Peak confidence scores in `[0.0, 1.0]`. |
| `labels` | `[B, K]` | `int64` | Foreground class IDs (`0` = `colony`). |
| `centers` | `[B, K, 2]` | `float32` | Sub-pixel centers `[cx, cy]` in input pixel space. |
| `radii` | `[B, K]` | `float32` | Predicted radial colony size `r_px` in pixels. |

*(Raw dense-head fallback also supports `heatmap [B, C, H/4, W/4]`, `offset [B, 2, H/4, W/4]`, `radius [B, 1, H/4, W/4]`.)*

---

### 2. 2D StarDist (`stardist`)
Performs star-convex instance segmentation along `N_rays` equiangular vectors (`theta_k = 2 * pi * k / N_rays`, default `N_rays = 32`).

**Primary Decoded Outputs:**
| Output Name | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `boxes` | `[B, K, 4]` | `float32` | Tight polygon bounding boxes `[xmin, ymin, xmax, ymax]`. |
| `scores` | `[B, K]` | `float32` | Object probability scores in `[0.0, 1.0]`. |
| `labels` | `[B, K]` | `int64` | Foreground class IDs (`0` = `colony`). |
| `centers` | `[B, K, 2]` | `float32` | Colony centers `[cx, cy]` in input pixel space. |
| `rays` | `[B, K, N_rays]` | `float32` | Predicted radial boundary distances along each ray. |
| `polygons` | `[B, K, N_rays, 2]` | `float32` | Polygon vertices `[x_k, y_k]` in input pixel space. |

*(Raw dense-head fallback also supports `prob [B, 1, H/4, W/4]` and `rays [B, N_rays, H/4, W/4]`.)*

---

### 3. Plate & Section Countability Classifier (`countability`)
Classifies whole dish images and/or 640×640 section tiles into `0: empty`, `1: countable`, or `2: uncountable`.

**Outputs:**
| Output Name | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `logits` | `[B, 3]` | `float32` | Unnormalized class logits `[empty, countable, uncountable]`. |
| `probs` | `[B, 3]` | `float32` | Softmax probabilities across the 3 classes. |
| `pred_class` | `[B]` | `int64` | Argmax class index (`0`, `1`, or `2`). |
| `is_countable` | `[B]` | `bool` | Binary countability flag derived from `P(uncountable) < threshold`. |

---

## Commercial Licensing Note
All model architectures and inference code supported by `autoCFU` are permissively licensed (Apache-2.0 / MIT / BSD-3-Clause) and safe for commercial deployment when paired with permissively or proprietarily trained ONNX weights. Models trained directly on the AGAR benchmark (`CC BY-NC 2.0`) remain restricted to non-commercial proof-of-concept evaluation.
