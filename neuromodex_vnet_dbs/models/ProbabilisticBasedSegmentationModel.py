from abc import abstractmethod

import numpy as np

import SimpleITK as sitk

from neuromodex_vnet_dbs.data.preprocessing import remove_outlier_intensities, denoise
from neuromodex_vnet_dbs.data.sitk_transform import get_float32_image_array
from neuromodex_vnet_dbs.models.SegmentationModelBase import SegmentationModelBase


class ProbabilisticBasedSegmentationModel(SegmentationModelBase):
    def __init__(self, n_components: int = 3, **kwargs):
        super().__init__(n_components=n_components, **kwargs)

    def get_sorted_label_volume(self, sitk_image) -> np.ndarray:
        image_array = get_float32_image_array(sitk_image)

        flat_image = image_array.flatten()
        flat_labels = np.argmax(self.labels, axis=1)

        class_means = {}
        for c in range(self.n_components):
            if np.any(flat_labels == c):
                class_means[c] = np.mean(flat_image[flat_image > 0][flat_labels == c])
            else:
                class_means[c] = np.inf

        sorted_classes = sorted(class_means, key=lambda k: class_means[k])

        remap_dict = {old: new + 1 for new, old in enumerate(sorted_classes)}
        remapped_labels = np.vectorize(remap_dict.get)(flat_labels)

        label_volume = np.zeros(image_array.shape)
        label_volume[image_array > 0] = remapped_labels

        return label_volume

    def plot_classes(self, sitk_image: sitk.Image):
        if self.labels is not None:
            self._log("Plotting classes")

            image_array = get_float32_image_array(sitk_image)
            label_mask = image_array > 0

            for n in range(self.n_components):
                tmp = np.zeros(image_array.shape)
                tmp[label_mask] = self.labels[:, n]

        else:
            self._log("Cannot Plot Classes. No labels found")

    def preprocess(self, sitk_image: sitk.Image) -> sitk.Image:
        sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
        sitk_image = denoise(sitk_image)
        sitk_image = remove_outlier_intensities(sitk_image)
        return sitk_image


    @abstractmethod
    def segment(self, sitk_image: sitk.Image):
        pass
