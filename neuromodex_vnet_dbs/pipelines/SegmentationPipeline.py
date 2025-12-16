import sys

import SimpleITK as sitk

from neuromodex_vnet_dbs.core.BaseProcessor import BaseProcessor

from neuromodex_vnet_dbs.data.postprocessing import combine_gmm_unet_volume
from neuromodex_vnet_dbs.data.preprocessing import roi_nonzero_slices, extract_roi
from neuromodex_vnet_dbs.data.sitk_image_handler import load_image
from neuromodex_vnet_dbs.models.CNNBasedSegmentationModel import CNNBasedSegmentationModel
from neuromodex_vnet_dbs.models.GMM import GMMSegmentationModel
from neuromodex_vnet_dbs.models.architecture.VNet import VNet
from neuromodex_vnet_dbs.data.postprocessing import post_process_filling


class SegmentationPipeline(BaseProcessor):

    def __init__(self, input_volume: str | sitk.Image, fill_empty=False, **kwargs):
        super().__init__(**kwargs)
        if type(input_volume) == str:
            input_volume = load_image(input_volume)

        # ROI extraction
        self.volume = extract_roi(input_volume, roi_nonzero_slices(input_volume))

        self.cnn_model_name = "SpacingAwareVNetS"
        self.fill_empty = fill_empty

    def segment_fast(self):
        try:
            self._log(f"Starting segmentation with model: {self.cnn_model_name}")

            model = CNNBasedSegmentationModel(VNet(spacing_aware=True), target_spacing=None, seg_name=self.cnn_model_name)

            # Preprocessing
            preprocessed_volume = model.preprocess(self.volume)

            # Segmentation
            model.segment(preprocessed_volume)

            # Save results
            segmentation = model.get_segmented_image(preprocessed_volume, self.volume)

            if self.fill_empty:
                mask = sitk.BinaryThreshold(self.volume, lowerThreshold=1, upperThreshold=1e10, insideValue=1,
                                outsideValue=0)
                segmentation = post_process_filling(segmentation, mask)

            self._log("CNN Segmentation completed successfully")
            return segmentation

        except Exception as e:
            self._log_error(f"Error while segmenting with model {self.cnn_model_name}: {str(e)}")
            sys.exit(1)

    def segment_gmm_csf(self):

        cnn_segmentation = self.segment_fast()

        try:
            model = GMMSegmentationModel()

            # Preprocessing
            preprocessed_volume = model.preprocess(self.volume)

            # Segmentation
            model.segment(preprocessed_volume)

            # Save results
            gmm_segmentation = model.get_segmented_image(preprocessed_volume, self.volume)

            self._log("GMM Segmentation completed successfully")
            if self.fill_empty:
                mask = sitk.BinaryThreshold(self.volume, lowerThreshold=1, upperThreshold=1e10, insideValue=1,
                                outsideValue=0)
                gmm_segmentation = post_process_filling(gmm_segmentation, mask)

            return combine_gmm_unet_volume(gmm_segmentation, cnn_segmentation)

        except Exception as e:
            self._log_error(f"Error while segmenting with GMM model: {str(e)}")
            sys.exit(1)
