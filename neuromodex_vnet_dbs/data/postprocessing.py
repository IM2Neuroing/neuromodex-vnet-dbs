import SimpleITK as sitk

from neuromodex_vnet_dbs.data.sitk_transform import reset_sitk_image_from_image_array, resample_to_spacing, \
    crop_to_match


def combine_gmm_unet_volume(gmm_sitk, unet_sitk):
    gmm_labels = sitk.GetArrayFromImage(gmm_sitk)
    unet_labels = sitk.GetArrayFromImage(unet_sitk)
    combined_labels = gmm_labels.copy()

    combined_labels[unet_labels == 2] = 2
    combined_labels[unet_labels == 3] = 3
    combined_labels[(gmm_labels == 3) & (unet_labels == 1)] = 2
    combined_labels[gmm_labels == 1] = 1
    return reset_sitk_image_from_image_array(gmm_sitk, combined_labels)


def post_process_filling(output_sitk, mask_sitk) -> sitk.Image:
    condition = sitk.And(mask_sitk == 1, output_sitk == 0)
    return sitk.Mask(output_sitk, condition == 0, outsideValue=1)


def resample_to_input_image_size_after_inference(output, preprocessed, original):
    segmented_image = sitk.GetImageFromArray(output.squeeze())
    segmented_image.CopyInformation(preprocessed)
    segmented_image = resample_to_spacing(segmented_image, original.GetSpacing())
    segmented_image = crop_to_match(segmented_image, original)
    segmented_image.CopyInformation(original)

    return segmented_image