Neuromodex VNet DBS
====================

Deep‑learning utilities for DBS workflows, including MRI brain tissue segmentation (VNet) and conductivity mapping. This repository provides a Python package and optional 3D Slicer modules to integrate the models into imaging workflows.

Features
--------
- VNet‑based multi‑class brain tissue segmentation
- Conductivity mapping utilities
- Pre/post‑processing with SimpleITK and SciPy
- PyTorch inference with automatic device selection (CPU/GPU)
- 3D Slicer plugin scaffolding for GUI‑based use

Installation
------------
Requirements: Python 3.9+

```
pip install neuromodex-vnet-dbs
```

The wheel bundles the `neuromodex_vnet_dbs/weights/` directory so the packaged models can load without any extra downloads. If you maintain your own weights, see the “Weights layout” section below.

Quick start
-----------
```python
import SimpleITK as sitk
from neuromodex_vnet_dbs import SegmentationPipeline

# Load an input image (e.g., NIfTI)
img = sitk.ReadImage("/path/to/volume.nii.gz")

# Run the segmentation pipeline
pipe = SegmentationPipeline()
result = pipe.run(img)

# The returned object and exact fields depend on your pipeline implementation.
# For example, to get a label volume as a SimpleITK image:
# label_img = result["label_image"]
```

Weights layout (packaged)
-------------------------
Runtime code expects weights under the installed package at:

```
neuromodex_vnet_dbs/weights/{seg_name}/
```

For example, a common file name is `best.pth`. During inference, `CNNBasedSegmentationModel` looks for the best checkpoint inside that folder, preferring `best.pth` when available, otherwise selecting the highest `epoch*.pth` checkpoint. If you ship custom weights, place them in the matching subfolder so that `self.seg_name` resolves to the correct directory at runtime.

Notes on model loading
----------------------
- The model auto‑selects CUDA when available, otherwise CPU.
- Checkpoints saved via DDP/DataParallel are handled (common prefixes like `module.` are stripped).
- If strict loading fails, it falls back to `strict=False` and logs missing/unexpected keys.

3D Slicer integration (optional)
--------------------------------
This repo includes helper scripts and example module folders under `slicer/`.

- To install one or more module folders into your local Slicer profile, run:

  ```
  python slicer/slicer_install_plugin.py
  ```

  Follow the prompts to choose the plugin(s) and target Slicer installation. Restart Slicer afterwards.

Build from source
-----------------
We use Hatchling to build the package.

```
pip install build hatchling
python -m build
```

Artifacts are written to `dist/` as a wheel and sdist. The build is configured to include `neuromodex_vnet_dbs/weights/**` in the wheel, so bundled weights are shipped with the package.

License
-------
This project is licensed under the terms of the MIT License. See the `LICENSE` file for details.

Citation
--------
If you use this project in your research, please cite the appropriate papers for VNet and any downstream methods you apply. Add your preferred citation format here.
