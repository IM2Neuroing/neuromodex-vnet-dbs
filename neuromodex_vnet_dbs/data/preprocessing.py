import SimpleITK as sitk
import numpy as np
from scipy.signal import find_peaks

from neuromodex_vnet_dbs.data.sitk_transform import get_float32_image_array, reset_sitk_image_from_image_array, \
    pad_to_size


def denoise(sitk_image) -> sitk.Image:
    denoised = sitk.CurvatureAnisotropicDiffusion(
        image1=sitk_image,
        timeStep=0.01,
        conductanceParameter=0.5,
        numberOfIterations=5
    )

    mask = sitk.BinaryThreshold(sitk_image, lowerThreshold=1, upperThreshold=1e10, insideValue=1,
                                outsideValue=0)
    denoised_masked = sitk.Mask(denoised, mask)

    return denoised_masked

def remove_outlier_intensities(sitk_image: sitk.Image):
    arr = get_float32_image_array(sitk_image)
    mask = arr > 0
    vals = arr[mask]

    # Build histogram
    hist, bin_edges = np.histogram(vals, bins=2048)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peaks, _ = find_peaks(hist, prominence=np.max(hist) * 0.1)

    if len(peaks) == 0:
        peak_idx = int(len(bin_centers) / 2)
    else:
        peak_idx = peaks[-1]

    peak_center = bin_centers[peak_idx]

    # Fit Gaussian to region around peak (± some range)
    window = (bin_centers > peak_center * 0.5) & (bin_centers < peak_center * 1.5)
    fit_vals = bin_centers[window]
    fit_hist = hist[window]
    mean = np.average(fit_vals, weights=fit_hist)
    std = np.sqrt(np.average((fit_vals - mean) ** 2, weights=fit_hist))
    cutoff = mean + 3 * std

    # Remove high-intensity tail voxels
    arr[(arr > cutoff) & mask] = 0

    return reset_sitk_image_from_image_array(sitk_image, arr)


def normalize_sitk(image: sitk.Image) -> sitk.Image:
    image_np = sitk.GetArrayFromImage(image).astype(np.float32)

    # Create mask for foreground
    mask = image_np > 0

    image_min = np.min(image_np[mask])
    image_max = np.max(image_np[mask])

    normalized_np = np.zeros_like(image_np, dtype=np.float32)
    normalized_np[mask] = (image_np[mask] - image_min) / (image_max - image_min)

    normalized_image = sitk.GetImageFromArray(normalized_np)
    normalized_image.CopyInformation(image)
    return normalized_image


def pad_to_divisible(sitk_image: sitk.Image, divisor: int = 16) -> sitk.Image:
    size = np.array(sitk_image.GetSize())
    target_size = ((size + divisor - 1) // divisor) * divisor
    return pad_to_size(sitk_image, target_size)


def roi_nonzero_slices(sitk_image: sitk.Image) -> tuple:
    mask = sitk.BinaryThreshold(sitk_image, lowerThreshold=0.1, upperThreshold=1e10)

    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(sitk_image, mask)
    bounding_box = stats.GetBoundingBox(1)  # 1 is the label value in the mask

    # Extract the min/max indices for each dimension
    min_x, max_x = bounding_box[0], bounding_box[1]
    min_y, max_y = bounding_box[2], bounding_box[3]
    min_z, max_z = bounding_box[4], bounding_box[5]

    # Add 1 to max coordinates because SimpleITK uses [min, max) convention
    max_x += 1
    max_y += 1
    max_z += 1

    roi = (int(min_x), int(min_y), int(min_z), int(max_x - min_x), int(max_y - min_y), int(max_z - min_z))

    return roi


def extract_roi(sitk_image: sitk.Image, roi: tuple) -> sitk.Image:
    index = roi[:3]
    size = roi[3:]
    roi_image = sitk.RegionOfInterest(sitk_image, size=size, index=index)
    new_origin = sitk_image.TransformIndexToPhysicalPoint(index)
    roi_image.SetOrigin(new_origin)
    return roi_image
