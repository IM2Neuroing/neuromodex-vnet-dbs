Neuromodex VNet DBS
====================

Deep‑learning utilities for DBS workflows, including MRI brain tissue segmentation (VNet) and conductivity mapping. This repository provides a Python package and optional 3D Slicer modules to integrate the models into imaging workflows.

Features
--------
- VNet‑based multi‑class brain tissue segmentation
- Conductivity mapping utilities
- Pre/post‑processing
- PyTorch inference with automatic device selection (CPU/GPU)
- 3D Slicer plugin scaffolding for GUI‑based use
- The segmentation model was trained using labels generated with **ELMA**, a semi‑automatic DBS tissue classification/segmentation tool (commonly used to classify tissues such as grey matter, white matter, blood, and CSF for patient‑specific DBS FEM modeling workflows). [[1]](https://liu.se/en/article/ne-downloads) [[2]](https://pubmed.ncbi.nlm.nih.gov/36086324/)

Installation
------------
Requirements: Python 3.9+

```
pip install neuromodex-vnet-dbs
```

The wheel bundles the `neuromodex_vnet_dbs/weights/` directory so the packaged models can load without any extra downloads.

Quick start Segmentation
-----------
```python
import SimpleITK as sitk
from neuromodex_vnet_dbs import SegmentationPipeline

# Load an input image (e.g., NIfTI)
img = sitk.ReadImage("/path/to/volume.nii.gz")
# or
img = "path/to/volume.nii.gz"

# Run the segmentation pipeline
pipe = SegmentationPipeline(img) # pass either as string or sitk volume
result = pipe.segment_fast(img) # ~7 seconds

# or this for clearer csf segmentation
result = pipe.segment_gmm_csf(img) # ~1.5 minutes

# The returned object is the segmented image
```

Quick Start Conductivity Mapping
-----------
```python
import SimpleITK as sitk
from neuromodex_vnet_dbs import ConductivityProcessingPipeline

mri_img = sitk.ReadImage("path/to/mri_image.nii.gz")
seg_img = sitk.ReadImage("path/to/seg_image.nii.gz")

pipe = ConductivityProcessingPipeline(seg_img, mri_img)
result = pipe.run()
```


3D Slicer integration
--------------------------------
This repo includes helper scripts and example module folders under `slicer/`.
These scripts can also be used as CLI tools.

- To install one or more module folders into your local Slicer profile, run:

  ```
  python slicer/package_slicer_modules.py
  python slicer/slicer_install_plugin.py
  ```

  Follow the prompts to choose the plugin(s) and target Slicer installation. Restart Slicer afterwards.

The plugins can then be found in Segmentation/BrainSegmentation and Electrical Conductivity/ConductivityMapping.

License
-------
This project is licensed under the terms of the MIT License. See the `LICENSE` file for details.

Citation
--------
If you use this project in your research, please cite the appropriate papers for VNet and any downstream methods you apply. Add your preferred citation format here.
