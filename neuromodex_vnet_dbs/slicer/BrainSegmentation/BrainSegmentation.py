#!/usr/bin/env python-real

import os
import sys
import argparse
import logging

# Windows path handling
if hasattr(sys, '_MEIPASS'):
    # PyInstaller bundle case
    sys.path.append(sys._MEIPASS)
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def install_if_missing(module, lib):
    try:
        __import__(module)
    except ImportError:
        try:
            import slicer.util
            slicer.util.pip_install(lib)
        except Exception:
            print("Please install " + lib + " manually.")


install_if_missing("torch", "torch")
install_if_missing("sklearn", "scikit-learn")
install_if_missing("scipy", "scipy")
install_if_missing("neuromodex-vnet-dbs", "neuromodex-vnet-dbs")

try:
    import SimpleITK as sitk
    from neuromodex_vnet_dbs import SegmentationPipeline
except ImportError as e:
    logging.error(f"Import error: {str(e)}")
    raise


def main():
    parser = argparse.ArgumentParser(
        description="Run MRI brain tissue segmentation using a pre-trained model.",
        epilog="Example: python script.py --inputVolume input.nii.gz --outputVolume output.nii.gz --model UNetMCombinedLabel"
    )
    parser.add_argument(
        "--inputVolume",
        required=True,
        help="Path to the input NIfTI image (e.g., 'input.nii.gz')."
    )
    parser.add_argument(
        "--outputVolume",
        required=True,
        help="Path where the output segmented image will be saved."
    )
    parser.add_argument(
        "--model",
        default="VNetS",
        choices=["VNetS", "VNetSGMM"],
        help="Segmentation model to use. Default is 'VNetS', Available: VNetS, VNetSGMM"
    )
    parser.add_argument(
        "--fillEmptyWithGM",
        default=False,
        help="Fill empty voxels with GM probability map.",
    )
    args = parser.parse_args()

    pipeline = SegmentationPipeline(
        args.inputVolume,
        args.fillEmptyWithGM,
    )

    if args.model == "VNetSGMM":
        seg = pipeline.segment_gmm_csf()
    else:
        seg = pipeline.segment_fast()

    sitk.WriteImage(seg, args.outputVolume)

if __name__ == "__main__":
    sys.exit(main())
