Installation & Setup
====================

This guide covers system prerequisites, installing AutoCFU, configuring ONNX Runtime execution providers (CPU vs. GPU), and setting up model checkpoints.

Prerequisites
-------------

* **Python**: Version 3.10, 3.11, or 3.12.
* **Lightweight Dependencies**: AutoCFU intentionally avoids heavy training frameworks such as PyTorch, TorchVision, or CUDA build toolchains. Its core runtime only requires:
  - ``numpy>=1.20.0``
  - ``onnxruntime>=1.14.0``
  - ``pillow>=9.0.0``
  - ``pyyaml>=6.0``

Installation Methods
--------------------

Using uv (Recommended)
^^^^^^^^^^^^^^^^^^^^^^

`uv <https://docs.astral.sh/uv/>`_ provides extremely fast environment resolution and installation:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/andreassag/autoCFU.git
   cd autoCFU

   # Create a virtual environment and install autoCFU
   uv venv .venv --python 3.12
   source .venv/bin/activate
   uv pip install -e .

Using pip
^^^^^^^^^

Standard installation using Python's built-in tools:

.. code-block:: bash

   git clone https://github.com/andreassag/autoCFU.git
   cd autoCFU

   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .

Developer & Documentation Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install developer testing tools (``pytest``, ``ruff``, ``ty``, ``pre-commit``) and documentation build tools (``sphinx``, ``furo``, ``sphinx-autodoc-typehints``):

.. code-block:: bash

   # Automated setup via Makefile:
   make install

   # Or manually with uv:
   uv pip install -e ".[dev,docs]"

   # Or manually with pip:
   pip install -e ".[dev,docs]"

Configuring Execution Providers: CPU vs. GPU
--------------------------------------------

By default, AutoCFU installs standard CPU ``onnxruntime``. If you have an NVIDIA GPU with CUDA installed and want GPU-accelerated inference:

1. Uninstall CPU ONNX Runtime:

   .. code-block:: bash

      pip uninstall onnxruntime

2. Install GPU-accelerated ONNX Runtime:

   .. code-block:: bash

      pip install onnxruntime-gpu

When running inference, pass ``--device cuda`` or leave ``--device auto`` (the default), which automatically selects ``CUDAExecutionProvider`` when available and gracefully falls back to ``CPUExecutionProvider``.

Model Artifacts
---------------

AutoCFU runs RT-DETRv4 models exported to ONNX format.

Default Model Location
^^^^^^^^^^^^^^^^^^^^^^

Place exported ONNX weights in the ``models/`` directory:

.. code-block:: text

   models/
   └── rtdetrv4_colony_detector.onnx

When running inference via the CLI, specify the model path using the required ``--model`` flag:

.. code-block:: bash

   autocfu --image data/dish.jpg --model models/rtdetrv4_colony_detector.onnx --save-overlay

Verifying Model Tensor Contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can verify that your ONNX model adheres to the expected tensor contract using Python:

.. code-block:: python

   import onnxruntime as ort

   session = ort.InferenceSession("models/rtdetrv4_colony_detector.onnx", providers=["CPUExecutionProvider"])

   # Inspect inputs
   print("Inputs:")
   for inp in session.get_inputs():
       print(f"  Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

   # Inspect outputs
   print("Outputs:")
   for out in session.get_outputs():
       print(f"  Name: {out.name}, Shape: {out.shape}, Type: {out.type}")

Expected signatures:
* **Inputs**: ``images`` (shape ``[batch_size, 3, 640, 640]``, float32) and optionally ``orig_target_sizes`` (shape ``[batch_size, 2]``, int64).
* **Outputs**: ``labels`` (int64), ``boxes`` (float32, ``[x1, y1, x2, y2]``), and ``scores`` (float32).
