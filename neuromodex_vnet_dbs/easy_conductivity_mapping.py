import SimpleITK as sitk

from neuromodex_vnet_dbs import ConductivityProcessingPipeline


def map_conductivities(input_seg: str | sitk.Image, input_mri: str | sitk.Image, csf=2, gm=0.123, wm=0.0754):
    cond_pipe = ConductivityProcessingPipeline(input_seg, input_mri, csf, gm, wm)
    return cond_pipe.run()
