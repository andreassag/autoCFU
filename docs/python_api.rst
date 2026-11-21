Python API Reference
====================

.. code-block:: python

   from autocfu import AutoCFUEngine, load_config

   cfg = load_config("configs/default_inference.yaml")
   engine = AutoCFUEngine("models/stardist.onnx", config=cfg)
   result = engine.predict("plate.jpg")
   print(result.colony_count, result.radii[:5])
   result.save_mask("outputs/plate_mask.tif")

.. toctree::
   :maxdepth: 2

   api/config
   api/slicer
   api/merger
   api/model
   api/pipeline
   api/cli
   api/utils
