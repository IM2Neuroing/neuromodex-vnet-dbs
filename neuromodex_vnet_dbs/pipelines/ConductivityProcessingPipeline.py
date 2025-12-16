import SimpleITK as sitk
from neuromodex_vnet_dbs.core.BaseProcessor import BaseProcessor
from neuromodex_vnet_dbs.data.sitk_image_handler import load_image
from neuromodex_vnet_dbs.data.sitk_transform import crop_to_match
from neuromodex_vnet_dbs.models.ConductivityMapper import ConductivityMapper


class ConductivityProcessingPipeline(BaseProcessor):
    def __init__(self, input_seg: str | sitk.Image, input_mri: str | sitk.Image, csf=2, gm=0.123, wm=0.0754, **kwargs):
        super().__init__(**kwargs)
        if type(input_seg) == str:
            input_seg = load_image(input_seg)
        if type(input_mri) == str:
            input_mri = load_image(input_mri)

        self.input_seg = input_seg
        self.input_mri = input_mri
        self.csf = csf
        self.gm = gm
        self.wm = wm

    def run(self):
        try:
            self._log("Starting conductivity mapping process...")

            self._log("Cropping MRI to match segmentation size if needed...")

            mri = crop_to_match(self.input_mri, self.input_seg)

            seg_array = sitk.GetArrayFromImage(self.input_seg)

            self._log("Running conductivity mapping...")
            cond_mapper = ConductivityMapper(self.csf, self.gm, self.wm, verbose=self.verbose)
            cond = cond_mapper.map_conductivities(mri, seg_array)

            self._log("Conductivity mapping complete.")
            return cond

        except Exception as e:
            self._log(f"Error during conductivity mapping: {str(e)}")
            return 1
