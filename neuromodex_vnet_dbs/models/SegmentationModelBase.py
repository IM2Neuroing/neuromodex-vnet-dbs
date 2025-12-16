import numpy as np
import SimpleITK as sitk
from abc import abstractmethod
from pathlib import Path

from neuromodex_vnet_dbs.core.BaseProcessor import BaseProcessor
from neuromodex_vnet_dbs.data.postprocessing import resample_to_input_image_size_after_inference


class SegmentationModelBase(BaseProcessor):
    """
    Abstract base to work with every segmentation model (Prob or CNN)
    """

    def __init__(self, seg_name: str, n_components: int = 3, verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(99)
        self.n_components = n_components
        self.verbose = verbose

        self.seg_name = seg_name

        # shape (WxDxH, n_components)
        self.labels = None

    @abstractmethod
    def get_sorted_label_volume(self, sitk_image) -> np.ndarray:
        pass

    @abstractmethod
    def preprocess(self, sitk_image: sitk.Image) -> sitk.Image:
        pass

    def save_segmented_image(self, processed_sitk: sitk.Image, original_sitk: sitk.Image, save_path=Path("output")):
        if self.labels is not None:
            self._log("Saving segmented image")
            label_img = self.get_segmented_image(processed_sitk, original_sitk)

            sitk.WriteImage(label_img,
                            f"{str(save_path / self.seg_name)}_segmentation.nii.gz")

        else:
            self._log_error("Saving segmented image failed: No labels found")

    def get_segmented_image(self, processed_sitk: sitk.Image, original_sitk: sitk.Image):
        if self.labels is not None:

            label_img = resample_to_input_image_size_after_inference(self.get_sorted_label_volume(processed_sitk),
                                                                     processed_sitk, original_sitk)
            return label_img
        else:
            self._log_error("Getting segmented image failed: No labels found")
        return None

    def post_process(self, sitk_image: sitk.Image, mask: sitk.Image):
        image_array = sitk.GetArrayFromImage(sitk_image)
        mask_array = sitk.GetArrayFromImage(mask)

        image_array[(mask_array == 1) & (image_array == 0)] = 1

        processed_image = sitk.GetImageFromArray(image_array)
        processed_image.CopyInformation(sitk_image)

        return processed_image

    @abstractmethod
    def segment(self, sitk_image: sitk.Image):
        pass
