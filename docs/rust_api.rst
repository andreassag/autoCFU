Rust Crate & CLI Guide
======================

AutoCFU ships a standalone Rust crate (``autocfu``) using ONNX Runtime (``ort``) with
strict mathematical parity against the Python engine (within :math:`10^{-6}` across
slicing windows, merging metrics, centroid/radius averaging, and StarDist 32-ray
polygon reconstruction).

Building & Testing
------------------

.. code-block:: bash

   cargo fmt --all -- --check
   cargo clippy --all-targets -- -D warnings
   cargo test --all-targets
   cargo doc --no-deps --open

Library Usage
-------------

.. code-block:: rust

   use autocfu::{AutoCFUEngine, EngineConfig};

   fn run() -> autocfu::Result<()> {
       let config = EngineConfig::default();
       let mut engine = AutoCFUEngine::new("models/stardist.onnx", config)?;
       let result = engine.predict_file("plate.jpg")?;
       result.save_mask("outputs/plate_mask.tif")?;
       result.save_json("outputs/plate.json")?;
       Ok(())
   }

Safety & Error Handling Guarantees
----------------------------------

* Crate targets Rust Edition 2021 with ``#![deny(unsafe_op_in_unsafe_fn)]`` and
  ``#![warn(missing_docs)]``.
* All fallible functions return ``autocfu::Result<T>`` backed by ``AutoCFUError``
  (via ``thiserror``).
* ``unwrap()`` and ``expect()`` are excluded from all production library and binary
  code paths.
