"""Dedicated preprocessing module for 3D printer defect detection (STFPM / PatchCore + Anomalib).

Provides unified preprocessing pipelines for both training (Torchvision/Anomalib DataModule)
and inference (OpenCV ndarray, file paths, and PyTorch tensors).

Pipeline Steps:
    1. 1:1 Square ROI Crop: [0, 250, 1000, 1250] (1000x1000) from 1200x1600 printer bed images.
       - Guarantees 1:1 geometric aspect ratio, eliminating anisotropic stretching of layer lines.
       - Completely excludes bottom hardware interference (camera mount / nozzle shade at y >= 1000).
    2. Native 3-Channel RGB Preservation:
       - Preserves full color spectrum without Grayscale or CLAHE noise amplification.
       - Perfectly conforms to ImageNet-pretrained ResNet-18/34 convolutional feature extractors.
    3. Antialiased Area Resize: decimation of 3-channel RGB to target resolution (512x512 / 256x256).
    4. ImageNet Normalization: standard mean and std scaling for pretrained convolutional backbones.
    5. Gatekeeper & Otsu Foreground Sanity Checks:
       - Rejects missing, corrupt, blank, or pre-print empty bed images (filament area < 5%).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors


# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default 1:1 square ROI crop for 1200x1600 images: [ymin, xmin, ymax, xmax] (1000x1000 square)
# Eliminates bottom hardware interference (y >= 1000) and matches 1:1 input aspect ratio
DEFAULT_ROI_CROP = (0, 250, 1000, 1250)
DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_MIN_FILAMENT_RATIO = 0.05  # Minimum 5% foreground area to confirm print presence


def check_filament_presence(
    image: np.ndarray,
    min_ratio: float = DEFAULT_MIN_FILAMENT_RATIO,
) -> Tuple[bool, float, np.ndarray]:
    """Evaluates whether an image contains print object using Otsu adaptive thresholding.

    Calculates the foreground pixel ratio without complex contour filtering,
    rejecting empty bed (pre-print) images while safely retaining small or early-stage parts.

    Args:
        image: Numpy image in BGR, RGB, or Grayscale format.
        min_ratio: Minimum foreground fraction required (default: 0.05 for 5%).

    Returns:
        Tuple of (is_present, foreground_ratio, binary_mask).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"Invalid image dimensions: {image.ndim}")

    if gray.dtype != np.uint8:
        if np.issubdtype(gray.dtype, np.floating):
            scale = 255.0 if gray.max() <= 1.0 else 1.0
            gray = np.clip(gray * scale, 0, 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)

    # Otsu thresholding automatically segments bright filament against dark PEI bed
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate simple foreground ratio
    foreground_ratio = float(np.mean(binary_mask > 0))
    is_present = bool(foreground_ratio >= min_ratio)

    return is_present, foreground_ratio, binary_mask


def validate_input_image(
    image_input: Any,
    min_dimension: int = 16,
    min_filament_ratio: Optional[float] = None,
) -> Tuple[bool, Optional[str], Optional[np.ndarray]]:
    """Gatekeeper validator checking for missing, corrupt, blank, or empty bed inputs.

    Args:
        image_input: File path (str or Path) or OpenCV numpy array.
        min_dimension: Minimum allowed width/height in pixels.
        min_filament_ratio: Optional minimum Otsu foreground ratio (e.g. 0.05).

    Returns:
        Tuple of (is_valid, error_message, loaded_bgr_image_array).
    """
    if image_input is None:
        return False, "Input image is None (missing input).", None

    # Handle file path input
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            return False, f"Image file does not exist: {img_path}", None
        if not img_path.is_file():
            return False, f"Path is not a regular file: {img_path}", None
        if img_path.stat().st_size == 0:
            return False, f"Image file is empty (0 bytes): {img_path}", None

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return False, f"Failed to decode image file (corrupted or unsupported format): {img_path}", None
        image_arr = img_bgr
    elif isinstance(image_input, torch.Tensor):
        # Convert torch.Tensor to numpy ndarray
        t = image_input.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3):  # (C, H, W) -> (H, W, C)
            t = t.permute(1, 2, 0)
        image_arr = t.numpy()
    elif isinstance(image_input, np.ndarray):
        image_arr = image_input
    else:
        return False, f"Unsupported input type: {type(image_input)}. Expected str, Path, np.ndarray, or torch.Tensor.", None

    # Check array properties
    if image_arr.size == 0:
        return False, "Image array is empty (size 0).", None
    if image_arr.ndim not in (2, 3):
        return False, f"Image array must have 2 or 3 dimensions, got ndim={image_arr.ndim}.", None

    h, w = image_arr.shape[:2]
    if h < min_dimension or w < min_dimension:
        return False, f"Image dimensions ({w}x{h}) are below minimum threshold ({min_dimension}x{min_dimension}).", None

    # Check for NaN or Inf in floating-point arrays
    if np.issubdtype(image_arr.dtype, np.floating) and not np.isfinite(image_arr).all():
        return False, "Image contains NaN or Inf values.", None

    # Check for degenerate constant/blank image
    if np.all(image_arr == 0):
        return False, "Image is completely pitch black (all pixels 0).", None

    # Check filament presence via Otsu threshold if configured
    if min_filament_ratio is not None and min_filament_ratio > 0.0:
        is_present, ratio, _ = check_filament_presence(image_arr, min_ratio=min_filament_ratio)
        if not is_present:
            return (
                False,
                f"Empty bed detected (filament presence {ratio*100:.1f}% is below required {min_filament_ratio*100:.1f}%).",
                image_arr,
            )

    return True, None, image_arr


def crop_roi_numpy(
    image: np.ndarray,
    roi_crop: Optional[Sequence[int]] = DEFAULT_ROI_CROP,
) -> np.ndarray:
    """Crops the region of interest from a numpy image.

    If roi_crop is specified and matches image bounds, performs [ymin:ymax, xmin:xmax].
    Otherwise, if the image is non-square, falls back to a 1:1 center square crop.

    Args:
        image: Numpy image of shape (H, W) or (H, W, C).
        roi_crop: Sequence of [ymin, xmin, ymax, xmax] or None.

    Returns:
        Cropped numpy array.
    """
    h, w = image.shape[:2]
    if roi_crop is not None and len(roi_crop) == 4:
        ymin, xmin, ymax, xmax = roi_crop
        if 0 <= ymin < ymax <= h and 0 <= xmin < xmax <= w:
            return image[ymin:ymax, xmin:xmax]

    # Fallback to 1:1 center square crop if aspect ratio is not 1:1
    if h != w:
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        return image[y_start : y_start + size, x_start : x_start + size]

    return image


def crop_roi_tensor(
    tensor: torch.Tensor,
    roi_crop: Optional[Sequence[int]] = DEFAULT_ROI_CROP,
) -> torch.Tensor:
    """Crops the region of interest from a PyTorch image or mask tensor.

    Args:
        tensor: Tensor of shape (..., H, W).
        roi_crop: Sequence of [ymin, xmin, ymax, xmax] or None.

    Returns:
        Cropped PyTorch tensor.
    """
    if tensor.ndim < 2:
        raise ValueError(f"Tensor must have at least 2 spatial dimensions (H, W), got {tensor.ndim}D tensor.")

    h, w = tensor.shape[-2], tensor.shape[-1]
    if roi_crop is not None and len(roi_crop) == 4:
        ymin, xmin, ymax, xmax = roi_crop
        if 0 <= ymin < ymax <= h and 0 <= xmin < xmax <= w:
            return tensor[..., ymin:ymax, xmin:xmax]

    # Fallback to 1:1 center square crop if aspect ratio is not 1:1
    if h != w:
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        return tensor[..., y_start : y_start + size, x_start : x_start + size]

    return tensor


def apply_spatial_patch_mask(
    anomaly_map: torch.Tensor,
    margin_ratio: float = 0.05,
) -> torch.Tensor:
    """Applies a spatial boundary mask to suppress false-positive edge artifacts.

    Args:
        anomaly_map: Anomaly map tensor of shape (B, 1, H, W) or (1, H, W) or (H, W).
        margin_ratio: Margin fraction to mask out from each border (default: 0.05).

    Returns:
        Anomaly map with peripheral boundary scores zeroed out.
    """
    if margin_ratio <= 0.0:
        return anomaly_map
    if margin_ratio >= 0.5:
        return torch.zeros_like(anomaly_map)

    h, w = anomaly_map.shape[-2], anomaly_map.shape[-1]
    pad_y = int(h * margin_ratio)
    pad_x = int(w * margin_ratio)

    if pad_y == 0 and pad_x == 0:
        return anomaly_map

    mask = torch.zeros((h, w), dtype=torch.bool, device=anomaly_map.device)
    mask[pad_y : h - pad_y, pad_x : w - pad_x] = True

    return anomaly_map * mask


class Printer3DTransform(transforms.Transform):
    """Torchvision v2 compatible transform pipeline for 3D printer anomaly detection.

    Preserves native 3-channel RGB fidelity without CLAHE or Grayscale distortion,
    enforcing 1:1 square crop geometry and standard ImageNet normalization.
    """

    def __init__(
        self,
        image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
        roi_crop: Optional[Sequence[int]] = DEFAULT_ROI_CROP,
        **kwargs: Any,  # Absorbs any legacy arguments gracefully
    ):
        super().__init__()
        self.image_size = tuple(image_size)
        self.roi_crop = tuple(roi_crop) if roi_crop is not None else None
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Applies transformation based on input tensor type."""
        del params  # Unused for deterministic transforms

        if isinstance(inpt, tv_tensors.Mask):
            # Mask preprocessing: 1:1 ROI crop -> nearest neighbor resize
            cropped = crop_roi_tensor(inpt, self.roi_crop)
            orig_ndim = cropped.ndim
            if orig_ndim == 2:
                cropped = cropped.unsqueeze(0)
            resized = F.resize(
                cropped,
                self.image_size,
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            if orig_ndim == 2:
                resized = resized.squeeze(0)
            return tv_tensors.Mask(resized)

        if isinstance(inpt, (tv_tensors.Image, torch.Tensor)):
            # Image preprocessing pipeline:
            # 1. 1:1 Square ROI Crop
            cropped = crop_roi_tensor(inpt, self.roi_crop)
            if cropped.ndim == 2:
                cropped = cropped.unsqueeze(0)

            # 2. Antialiased area resize on native 3-channel RGB
            resized_3ch = F.resize(
                cropped,
                self.image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )

            # Ensure 3 channels
            if resized_3ch.shape[-3] == 1:
                resized_3ch = resized_3ch.repeat_interleave(3, dim=-3)

            # 3. Standard ImageNet normalization on 3-channel RGB
            normalized = self.normalize(resized_3ch)

            if isinstance(inpt, tv_tensors.Image):
                return tv_tensors.Image(normalized)
            return normalized

        return inpt


class Printer3DPreprocessor:
    """Unified preprocessor for training and standalone inference execution."""

    def __init__(
        self,
        image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
        roi_crop: Optional[Sequence[int]] = DEFAULT_ROI_CROP,
        device: Optional[Union[str, torch.device]] = None,
        min_filament_ratio: Optional[float] = DEFAULT_MIN_FILAMENT_RATIO,
        **kwargs: Any,  # Absorbs legacy arguments gracefully
    ):
        self.image_size = tuple(image_size)
        self.roi_crop = tuple(roi_crop) if roi_crop is not None else None
        self.min_filament_ratio = min_filament_ratio

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._transform = Printer3DTransform(
            image_size=self.image_size,
            roi_crop=self.roi_crop,
        )

    def get_torchvision_transform(self) -> Printer3DTransform:
        """Returns the torchvision v2 transform suitable for Anomalib Folder DataModule."""
        return self._transform

    def preprocess_array(
        self,
        img_arr: np.ndarray,
        is_bgr: bool = True,
    ) -> torch.Tensor:
        """Transforms a validated numpy image array using Printer3DTransform directly.

        Args:
            img_arr: Validated numpy array of shape (H, W, 3) or (H, W).
            is_bgr: True if 3-channel input is BGR format (cv2 default).

        Returns:
            torch.Tensor of shape (1, 3, H, W) on self.device.
        """
        if img_arr.ndim == 2:
            tensor = torch.from_numpy(img_arr.copy()).unsqueeze(0).float()
        elif img_arr.ndim == 3:
            if img_arr.shape[2] == 3:
                arr_rgb = img_arr[:, :, ::-1].copy() if is_bgr else img_arr.copy()
                tensor = torch.from_numpy(arr_rgb).permute(2, 0, 1).float()
            elif img_arr.shape[2] == 1:
                tensor = torch.from_numpy(img_arr[:, :, 0].copy()).unsqueeze(0).float()
            else:
                raise ValueError(f"Unsupported number of channels: {img_arr.shape[2]}")
        else:
            raise ValueError(f"Unsupported array dimensions: {img_arr.ndim}")

        if not np.issubdtype(img_arr.dtype, np.floating) or img_arr.max() > 1.0:
            tensor = tensor / 255.0

        transformed = self._transform(tv_tensors.Image(tensor))
        return transformed.unsqueeze(0).to(self.device)

    def preprocess_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        is_bgr: bool = True,
        check_filament: bool = False,
    ) -> torch.Tensor:
        """Preprocesses an input image into a normalized tensor ready for model inference.

        Args:
            image_input: Path to image file (str or Path) or OpenCV numpy array.
            is_bgr: Whether numpy input is in BGR format (True for cv2.imread).
            check_filament: Whether to enforce Otsu 5% filament presence verification.

        Returns:
            torch.Tensor of shape (1, 3, H, W) on the configured device.
        """
        min_ratio = self.min_filament_ratio if check_filament else None
        is_valid, err_msg, img_arr = validate_input_image(
            image_input,
            min_filament_ratio=min_ratio,
        )
        if not is_valid or img_arr is None:
            raise ValueError(f"Gatekeeper validation failed: {err_msg}")

        return self.preprocess_array(img_arr, is_bgr=is_bgr)
