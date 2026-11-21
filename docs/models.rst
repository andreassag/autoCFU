Supported ONNX Models
=====================

``autoCFU`` automatically inspects ONNX graph output names (or custom metadata
``autocfu_model_type``) to select the right execution path, or accepts an explicit
``--model-type`` override.

Common Input Tensor
-------------------

* **``images``**: ``float32 [batch_size, 3, H, W]`` — Normalized RGB tensor in
  ``[0.0, 1.0]`` (default ``H = W = 640``).

Model 1: Radial CenterNet (``centernet``)
-----------------------------------------

Detects colony centers and radii, mapping tile coordinates back into global Petri
dish space and deduplicating across seams with ``center_dist``, ``ios``, or ``iou``.

**Outputs:**

* ``boxes`` — ``float32 [B, K, 4]`` (``[xmin, ymin, xmax, ymax]``)
* ``scores`` — ``float32 [B, K]``
* ``labels`` — ``int64 [B, K]``
* ``centers`` — ``float32 [B, K, 2]``
* ``radii`` — ``float32 [B, K]``

Raw dense map fallback also supports ``heatmap``, ``offset``, and ``radius``.

Model 2: 2D StarDist (``stardist``)
-----------------------------------

Predicts object center probabilities and radial distances along ``N_rays = 32``
equiangular vectors, reconstructing star-convex polygons and rasterizing 16-bit
``uint16`` instance segmentation masks via ``--save-mask``.

**Outputs:**

* ``boxes`` — ``float32 [B, K, 4]``
* ``scores`` — ``float32 [B, K]``
* ``labels`` — ``int64 [B, K]``
* ``centers`` — ``float32 [B, K, 2]``
* ``rays`` — ``float32 [B, K, 32]``
* ``polygons`` — ``float32 [B, K, 32, 2]``

Raw dense map fallback also supports ``prob`` and ``rays``.

Model 3: Countability Classifier (``countability``)
---------------------------------------------------

Classifies a dish or ``640×640`` dish section into ``0: empty``, ``1: countable``,
or ``2: uncountable``. Binary countability is evaluated via:

.. code-block:: text

   is_countable = P(uncountable) < uncountable_threshold

**Outputs:**

* ``logits`` — ``float32 [B, 3]``
* ``probs`` — ``float32 [B, 3]``
* ``pred_class`` — ``int64 [B]``
* ``is_countable`` — ``bool [B]``
