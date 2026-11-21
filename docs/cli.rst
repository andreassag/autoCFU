Command-Line Interface
======================

The Python CLI (``autocfu``, ``autocfu-infer``, ``agar-infer``) and native Rust CLI
(``./target/release/autocfu``) share identical command-line flags and YAML schemas.

Examples
--------

Radial CenterNet colony detection:

.. code-block:: bash

   autocfu --image plate.jpg --model models/centernet.onnx \
     --save-overlay --save-json --save-csv --output-dir outputs/

2D StarDist segmentation with ``uint16`` instance mask export:

.. code-block:: bash

   autocfu --image plate.jpg --model models/stardist.onnx \
     --save-overlay --save-mask --save-json --output-dir outputs/

Plate and section countability screening:

.. code-block:: bash

   autocfu --image plate.jpg --model models/countability.onnx \
     --countability-scope both --uncountable-threshold 0.50 \
     --save-overlay --save-json --save-csv --output-dir outputs/

CLI Options
-----------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Flag
     - Description
   * - ``--image`` / ``--input-dir``
     - Single dish image or batch directory.
   * - ``--model``
     - Path to exported ``.onnx`` weights.
   * - ``--model-type``
     - ``auto``, ``centernet``, ``stardist``, or ``countability``.
   * - ``--config``
     - Optional YAML configuration file path.
   * - ``--confidence``
     - Minimum score threshold in ``[0.0, 1.0]``.
   * - ``--slice-size`` / ``--overlap``
     - Sliding-window tile size and fractional overlap.
   * - ``--merger`` / ``--match-metric``
     - ``greedy_nmm``, ``nms``, ``none`` and ``center_dist``, ``ios``, ``iou``.
   * - ``--match-threshold``
     - Threshold applied by cross-tile merger.
   * - ``--min-radius`` / ``--max-radius``
     - Valid colony radius bounds in pixels.
   * - ``--countability-scope``
     - ``both``, ``plate``, or ``sections``.
   * - ``--uncountable-threshold``
     - Probability threshold on ``uncountable`` class.
   * - ``--save-overlay``
     - Save rendered colony or countability visualizations.
   * - ``--save-mask``
     - Save 16-bit ``uint16`` instance masks (CenterNet / StarDist).
   * - ``--save-json`` / ``--save-csv``
     - Export per-image predictions and batch CSV summaries.
   * - ``--output-dir``
     - Output directory for rendered overlays, masks, JSON, and CSV exports.
   * - ``--show-labels``
     - Annotate colony detections with confidence scores on saved overlays.
   * - ``--device``
     - Execution provider preference (``auto``, ``cpu``, ``cuda``).
