from typing import Tuple

import SimpleITK as sitk
import numpy as np


def get_float32_image_array(sitk_image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk_image).astype(np.float32)

def reset_sitk_image_from_image_array(sitk_image: sitk.Image, image_array: np.ndarray) -> sitk.Image:
    new = sitk.GetImageFromArray(image_array)
    new.CopyInformation(sitk_image)
    return new


def resample_to_spacing(sitk_image: sitk.Image,
                        target_spacing: Tuple[float, float, float],
                        interpolator: int = sitk.sitkNearestNeighbor) -> sitk.Image:
    old_spacing = np.array(sitk_image.GetSpacing())
    old_size = np.array(sitk_image.GetSize())
    new_size = np.round(old_size * (old_spacing / np.array(target_spacing))).astype(int).tolist()

    # Use original origin and direction
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetInterpolator(interpolator)
    resample.SetTransform(sitk.Transform())  # identity
    resampled_img = resample.Execute(sitk_image)

    return resampled_img


def pad_to_size(sitk_image: sitk.Image, target_size) -> sitk.Image:
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    origin = np.array(sitk_image.GetOrigin())
    target_size = np.array(target_size)

    if np.any(target_size < size):
        raise ValueError(
            f"Target size {target_size} must be >= image size {size} in all dimensions."
        )

    pad_before = ((target_size - size) // 2).astype(int)
    pad_after = (target_size - size - pad_before).astype(int)

    padded_image = sitk.ConstantPad(
        sitk_image,
        padLowerBound=pad_before.tolist(),
        padUpperBound=pad_after.tolist(),
    )

    # Adjust the origin so the image stays centered in physical space
    new_origin = origin - pad_before * spacing
    padded_image.SetOrigin(tuple(new_origin))

    return padded_image


def crop_to_match(larger_image, reference_image):
    # Get geometry
    ls, lo, lsp = larger_image.GetSize(), larger_image.GetOrigin(), larger_image.GetSpacing()
    ss, so, ssp = reference_image.GetSize(), reference_image.GetOrigin(), reference_image.GetSpacing()

    # Check that spacing matches (within small tolerance)
    if not np.allclose(lsp, ssp, rtol=1e-4, atol=1e-5):
        raise ValueError(f"Spacing mismatch: larger={lsp}, smaller={ssp}")

    # Compute voxel offset between origins
    offset_world = np.array(so) - np.array(lo)
    offset_voxels = np.round(offset_world / np.array(lsp)).astype(int)

    # Compute crop region
    start_index = np.maximum(offset_voxels, 0)
    end_index = np.minimum(np.array(ls), start_index + np.array(ss))
    size = (end_index - start_index).tolist()

    if any(s <= 0 for s in size):
        raise ValueError("Images do not overlap or crop region is invalid!")

    # Perform crop
    cropped = sitk.RegionOfInterest(larger_image, size=size, index=start_index.tolist())

    # Adjust origin to exactly match the smaller image
    cropped.SetOrigin(reference_image.GetOrigin())
    cropped.SetDirection(reference_image.GetDirection())

    return cropped