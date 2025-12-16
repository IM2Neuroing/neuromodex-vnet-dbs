#!/usr/bin/env python-real

import os
import sys
import logging
import argparse

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


install_if_missing("neuromodex-vnet-dbs", "neuromodex-vnet-dbs")


try:
    import SimpleITK as sitk
    from neuromodex_vnet_dbs import ConductivityProcessingPipeline
except ImportError as e:
    logging.error(f"Import error: {str(e)}")
    raise


def main():
    parser = argparse.ArgumentParser(description="Slicer CLI Plugin: Conductivity Mapping")

    # Slicer-compatible CLI parameters
    parser.add_argument("--inputSegmentation", required=True, help="Input segmentation volume (e.g. NIfTI).")
    parser.add_argument("--inputMRI", required=True, help="Input MRI volume (e.g. NIfTI).")
    parser.add_argument("--outputVolume", required=True, help="Path to save the resulting output volume (e.g. NIfTI file)")

    # Optional parameters
    parser.add_argument("--csfConductivity", type=float, default=2.0, help="Conductivity of CSF (default: 2.0)")
    parser.add_argument("--gmConductivity", type=float, default=0.123, help="Conductivity of GM (default: 0.123)")
    parser.add_argument("--wmConductivity", type=float, default=0.0754, help="Conductivity of WM (default: 0.0754)")

    args = parser.parse_args()

    pipeline = ConductivityProcessingPipeline(
        input_seg=args.inputSegmentation,
        input_mri=args.inputMRI,
        csf=args.csfConductivity,
        gm=args.gmConductivity,
        wm=args.wmConductivity
    )

    cond = pipeline.run()
    sitk.WriteImage(cond, args.outputVolume)

if __name__ == "__main__":

    sys.exit(main())
