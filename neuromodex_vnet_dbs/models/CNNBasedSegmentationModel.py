from pathlib import Path

import numpy as np
import torch
import SimpleITK as sitk

from scipy.ndimage import binary_erosion

from neuromodex_vnet_dbs.data.preprocessing import denoise, remove_outliers_if_contrast_agent, \
    remove_outlier_intensities, normalize_sitk, pad_to_divisible
from neuromodex_vnet_dbs.data.sitk_transform import get_float32_image_array, resample_to_spacing
from neuromodex_vnet_dbs.models.SegmentationModelBase import SegmentationModelBase


class CNNBasedSegmentationModel(SegmentationModelBase):
    """
    Modular base class for CNN based segmentation models.
    """

    def __init__(self, model, padding_divisibility=16, target_spacing=(0.625, 0.625, 1.3),
                 n_components: int = 4, **kwargs):

        super().__init__(n_components=n_components, **kwargs)

        self.padding_divisibility = padding_divisibility
        self.target_spacing = target_spacing

        package_dir = Path(__file__).parent.parent

        if not (package_dir / "weights").exists():
            raise ValueError(f"Weights not found in {package_dir / 'weights'}")

        self.weight_output_path = Path(package_dir / f"weights/{self.seg_name}")

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess(self, sitk_image: sitk.Image):

        sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
        sitk_image.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
        if self.target_spacing is not None:
            sitk_image = resample_to_spacing(sitk_image, target_spacing=self.target_spacing, interpolator=sitk.sitkLinear)
        sitk_image = denoise(sitk_image)
        sitk_image = remove_outliers_if_contrast_agent(sitk_image)
        sitk_image = remove_outlier_intensities(sitk_image)
        sitk_image = normalize_sitk(sitk_image)
        sitk_image = pad_to_divisible(sitk_image, divisor=self.padding_divisibility)

        return sitk_image

    def segment(self, sitk_image: sitk.Image):
        self._log("Starting segmentation")
        spacing = torch.tensor(sitk_image.GetSpacing())
        image = np.expand_dims(sitk.GetArrayFromImage(sitk_image), axis=0)
        image = torch.from_numpy(image)
        self.load_state_dict()
        self.model.eval()

        with torch.no_grad():

            if self.target_spacing is None:
                output = self.model(image.unsqueeze(0).to(self.device), spacing.unsqueeze(0).to(self.device))
            else:
                output = self.model(image.unsqueeze(0).to(self.device))

        output = output.cpu()

        probs = torch.softmax(output, dim=1)

        self.labels = probs.numpy().reshape(4, -1).T
        self._log("Segmentation finished")

    def load_state_dict(self):

        best_full_name = ""
        best_full_epoch = -1

        if self.weight_output_path.exists():
            if self.weight_output_path / "best.pth":
                best_full_name = "best.pth"
            else:
                for weight in self.weight_output_path.iterdir():
                    if weight.is_file():

                        name = weight.name
                        try:
                            current_epoch = int(name.split("epoch")[1].split(".")[0])
                        except Exception:
                            # skip files that don't match expected naming
                            continue

                        if current_epoch > best_full_epoch:
                            best_full_epoch = current_epoch
                            best_full_name = name

        if best_full_name == "":
            self._log_error("No weights found")
            return

        weight_path = str(self.weight_output_path / best_full_name)
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)

        state_dict = None
        if isinstance(checkpoint, dict):
            for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break
            if state_dict is None:
                # assume it's already a state dict
                state_dict = checkpoint
        else:
            self._log_error(f"Unsupported checkpoint format in {weight_path}")
            return

        # Move model to device before loading
        self.model.to(self.device)

        # Try strict load first, fallback to strict=False
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            self._log(f"Strict load failed with: {e}. Retrying with strict=False.")
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if len(missing) > 0:
                self._log(f"Missing keys when loading weights: {missing}")
            if len(unexpected) > 0:
                self._log(f"Unexpected keys when loading weights: {unexpected}")

    def get_sorted_label_volume(self, sitk_image: sitk.Image) -> np.ndarray:
        image_array = get_float32_image_array(sitk_image)

        w, h, d = image_array.shape

        brain_mask_3d = image_array > 0
        eroded_brain_mask = binary_erosion(brain_mask_3d, iterations=2)
        brain_mask = eroded_brain_mask.flatten()

        # Only suppress background where we are confident it's brain
        self.labels[:, 0][brain_mask] = 0

        flat_labels = np.argmax(self.labels, axis=1)

        label_volume = flat_labels.reshape(w, h, d)

        return label_volume
