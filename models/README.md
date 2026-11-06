# Models Directory

This directory is the designated local storage path for trained model weights and exported ONNX runtime artifacts used by the inference engine (`src/autocfu/`).

> [!NOTE]
> Large weight files (`*.onnx`, `*.pth`, `*.pt`, `*.bin`) are excluded from version control via `.gitignore`.
> Do not commit binary model checkpoints to the repository.

---

## Expected Artifacts

| Model File | Backbone | Resolution | Queries | Format |
| :--- | :--- | :--- | :--- | :--- |
| `rtdetrv2_r18vd_agar_640.onnx` | ResNet-18-vd | 640×640 | 300 | ONNX (opset 16+) |
| `rtdetrv2_r34vd_agar_640.onnx` | ResNet-34-vd | 640×640 | 300 | ONNX (opset 16+) |

---

## ONNX Model Tensor Signatures

The inference engine expects RT-DETRv2 models exported with the following standard signatures:

### Inputs
1. **`images`**:
   - Shape: `[batch_size, 3, 640, 640]`
   - Type: `float32`
   - Value range: `[0.0, 1.0]` (RGB channel order, CHW layout)
2. **`orig_target_sizes`**:
   - Shape: `[batch_size, 2]`
   - Type: `int64`
   - Content: Image height and width for coordinate scaling: `[orig_h, orig_w]`, typically `[[640, 640]]` for 640×640 tile slices.

### Outputs
1. **`labels`**:
   - Shape: `[batch_size, 300]`
   - Type: `int64`
   - Content: Predicted category IDs (class `0` corresponds to `colony`).
2. **`boxes`**:
   - Shape: `[batch_size, 300, 4]`
   - Type: `float32`
   - Content: Predicted bounding box coordinates in `[x1, y1, x2, y2]` pixel format.
3. **`scores`**:
   - Shape: `[batch_size, 300]`
   - Type: `float32`
   - Content: Confidence probability scores in `[0.0, 1.0]`.

---

## How to Generate Model Weights

Model training and export pipelines are maintained in the companion repository [autoCFU-training](https://github.com/andreassag/autoCFU-training):

1. Follow the training workflow in `autoCFU-training` (via `notebooks/02-train-RT-DETRv2-colony-detection.ipynb` in Google Colab or via `scripts/train.py`).
2. Run fine-tuning on the 640×640 tiled AGAR dataset.
3. Export the trained checkpoint to ONNX (`RTDETR.export()`).
4. Place the resulting `.onnx` file in this `models/` directory:
   ```bash
   cp path/to/rtdetrv2_r18vd_agar_640.onnx models/
   ```


---

## Compliance Notice

Model weights fine-tuned on the AGAR dataset inherit the **Creative Commons Attribution-NonCommercial 2.0 Generic (CC BY-NC 2.0)** license terms. These weights are restricted to educational, non-commercial, and academic research purposes only. Refer to `NOTICE` and `licenses/LICENSE-AGAR-DATASET.txt` for details.
