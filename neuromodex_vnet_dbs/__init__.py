"""Top-level package for neuromodex_vnet_dbs.

This package provides segmentation and conductivity mapping utilities
for DBS workflows.
"""

__all__ = [
    "SegmentationPipeline",
    "ConductivityProcessingPipeline",
]

__version__ = "0.1.0"

# Re-export common entry points (lightweight imports only)
from .SegmentationPipeline import SegmentationPipeline  # noqa: E402,F401
from .ConductivityProcessingPipeline import ConductivityProcessingPipeline  # noqa: E402,F401
