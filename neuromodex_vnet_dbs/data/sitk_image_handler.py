import SimpleITK as sitk
from pathlib import Path


def load_image(path: Path | str):
    if isinstance(path, str):
        path = Path(path)
    if path.is_dir():
        return _load_dicom(str(path))
    elif path.name.endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")):
        if Path(path).is_file():
            return sitk.ReadImage(path, outputPixelType=sitk.sitkFloat32)
        else:
            raise FileNotFoundError(f"File does not exist: {str(path)}")
    else:
        raise ValueError(f"Unsupported image format or path: {str(path)}")


def _load_dicom(dicom_dir: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()

    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)

    if not series_ids:
        raise FileNotFoundError(f"No DICOM Series in {dicom_dir} found")

    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])

    reader.SetFileNames(series_file_names)
    return reader.Execute()


def save_image(image: sitk.Image, path: str):
    sitk.WriteImage(image, path)
