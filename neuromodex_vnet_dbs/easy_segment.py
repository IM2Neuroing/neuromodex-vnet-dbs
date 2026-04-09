from neuromodex_vnet_dbs import SegmentationPipeline
import SimpleITK as sitk


def segment_vnet(input_volume: str | sitk.Image, fill_empty=False):
    seg_pipeline = SegmentationPipeline(input_volume, fill_empty=fill_empty)
    return seg_pipeline.segment_fast()


def segment_vnet_gmm(input_volume: str | sitk.Image, fill_empty=False):
    seg_pipeline = SegmentationPipeline(input_volume, fill_empty=fill_empty)
    return seg_pipeline.segment_gmm_csf()
