import SimpleITK as sitk
import numpy as np
from collections import deque
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

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


def find_flat_tail_cutoff(bin_centers, hist, peak_idx):
    # 1) Smooth histogram heavily to estimate slope/curvature
    smooth = gaussian_filter1d(hist.astype(float), sigma=3)

    # 2) First & second derivatives
    dH = np.gradient(hist)
    ddH = np.gradient(dH)

    # 3) Define thresholds (tuneable)
    slope_thresh = hist.max() * 1e-8  # low slope
    curve_thresh = hist.max() * 1e-8  # low curvature
    height_thresh = hist.max() * 0.5  # low-amplitude tail tissue

    # 4) Search only to the RIGHT of the selected peak
    for i in range(peak_idx + 1, len(hist)):
        if (abs(dH[i]) < slope_thresh and
                abs(ddH[i]) < curve_thresh and
                smooth[i] < height_thresh):
            return bin_centers[i]

    # fallback (should rarely trigger)
    return bin_centers[-1]


def remove_outliers_if_contrast_agent(sitk_image: sitk.Image):
    arr = get_float32_image_array(sitk_image)
    mask = arr > 0
    vals = arr[mask]

    # Build histogram
    hist, bin_edges = np.histogram(vals, bins=2048)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peaks, _ = find_peaks(hist, prominence=np.max(hist) * 0.1)
    peak_idx = peaks[-1]

    cutoff = find_flat_tail_cutoff(bin_centers, hist, peak_idx)

    if len(arr[(arr > cutoff) & mask]) / len(  # todo check before publication or just vague
            arr.flatten()) > 0.005:
        print(cutoff)
        return robust_dynamic_seed_outlier_removal_region_fill(sitk_image, min_intensity=cutoff)

    return sitk_image


def remove_outlier_intensities(sitk_image: sitk.Image):
    arr = get_float32_image_array(sitk_image)
    mask = arr > 0
    vals = arr[mask]

    # Build histogram
    hist, bin_edges = np.histogram(vals, bins=2048)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peaks, _ = find_peaks(hist, prominence=np.max(hist) * 0.1)
    # peak_idx = np.argmax(hist)
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


def robust_dynamic_seed_outlier_removal_region_fill(
        image: sitk.Image,
        base_tol: float = 100.0,
        tol_growth: float = 0.3,
        growth_step: int = 50,
        max_tol: float = 400.0,
        ratio_tol: float = 0.7,

        # global stopping rule
        min_intensity: float = None,
        max_iterations: int = 500
):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)

    # auto config

    base_tol = arr.max() * 0.01
    max_tol = arr.max() / 4

    mask = np.zeros_like(arr, dtype=bool)

    # if not given, stop at 30% of global max intensity
    if min_intensity is None:
        min_intensity = arr.max() * 0.30

    neigh = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    iteration = 0

    first_seed_val = arr.max()

    while iteration < max_iterations:

        # -------------------------------
        # 1) Select brightest available voxel
        # -------------------------------
        remaining = np.where(~mask, arr, -np.inf)
        seed_val = remaining.max()

        # stop when no significant bright areas remain
        if seed_val < min_intensity:
            break

        if seed_val < first_seed_val * 0.5:
            break

        seed = np.unravel_index(np.argmax(remaining), arr.shape)
        iteration += 1

        visited = np.zeros_like(arr, dtype=bool)
        q = deque([seed])

        tolerance = base_tol
        count = 0

        # -------------------------------
        # 2) Aggressive flood fill
        # -------------------------------
        while q:
            z, y, x = q.popleft()
            if visited[z, y, x]:
                continue

            visited[z, y, x] = True
            mask[z, y, x] = True
            count += 1

            if count % growth_step == 0:
                tolerance = min(tolerance * (1 + tol_growth), max_tol)

            for dz, dy, dx in neigh:
                nz, ny, nx = z + dz, y + dy, x + dx

                if not (0 <= nz < arr.shape[0] and
                        0 <= ny < arr.shape[1] and
                        0 <= nx < arr.shape[2]):
                    continue

                if visited[nz, ny, nx] or mask[nz, ny, nx]:
                    continue

                nval = arr[nz, ny, nx]

                # acceptance rules:
                # A) absolute closeness to seed
                abs_accept = abs(nval - seed_val) < tolerance

                # B) relative ratio to seed
                ratio_accept = (nval / seed_val) > ratio_tol if seed_val > 0 else False

                if abs_accept or ratio_accept:
                    q.append((nz, ny, nx))

    # -------------------------------
    # 3) Apply mask → remove regions
    # -------------------------------
    arr[mask] = 0
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


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
