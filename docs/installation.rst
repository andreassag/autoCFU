Installation
============

Python Package
--------------

AutoCFU targets Python 3.10+ and excludes heavyweight training frameworks from its
runtime footprint.

Using ``uv`` (recommended):

.. code-block:: bash

   uv venv .venv
   source .venv/bin/activate
   uv pip install -e .

Using standard ``pip``:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .

Development, linting, testing, and documentation extras:

.. code-block:: bash

   uv pip install -e ".[dev,docs]"

Rust Binary & Crate
-------------------

AutoCFU includes a native Rust crate (``autocfu``) providing mathematically identical
slicing, merging, ONNX inference, and 16-bit TIFF/PNG mask export.

.. code-block:: bash

   cargo build --release
   cargo test --all-targets

To enable CUDA GPU execution in Rust:

.. code-block:: bash

   cargo build --release --features cuda
