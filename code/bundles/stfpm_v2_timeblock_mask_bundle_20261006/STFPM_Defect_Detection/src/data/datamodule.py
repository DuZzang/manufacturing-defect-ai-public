"""DataModule wrapper for 3D printer anomaly detection dataset."""

from pathlib import Path
from typing import Any, Optional, Sequence, Union

from anomalib.data import Folder

from src.data.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_ROI_CROP,
    Printer3DTransform,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def build_datamodule(
    root: Union[str, Path] = "data/printer3d",
    normal_dir: str = "train/good",
    normal_test_dir: Optional[str] = None,
    abnormal_dir: Optional[str] = "test/defect",
    mask_dir: Optional[str] = None,
    normal_split_ratio: Optional[float] = None,
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
    roi_crop: Optional[Sequence[int]] = DEFAULT_ROI_CROP,
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
    num_workers: int = 4,
    **kwargs: Any,  # Absorbs any legacy configuration parameters gracefully
) -> Folder:
    """Builds an Anomalib Folder DataModule with integrated 1:1 ROI crop and native RGB pipeline.

    Args:
        root: Root directory containing image subdirectories.
        normal_dir: Directory containing normal images for training.
        normal_test_dir: Directory containing normal images for testing (or None to auto-split).
        abnormal_dir: Directory containing defect images for testing.
        mask_dir: Directory containing ground truth pixel masks (optional).
        normal_split_ratio: Fraction of normal images reserved for test/val when normal_test_dir is None.
        image_size: Image resize dimensions [H, W].
        roi_crop: 1:1 square region of interest crop [ymin, xmin, ymax, xmax] or None.
        train_batch_size: Batch size during training (<= 16 for GTX 1660 SUPER).
        eval_batch_size: Batch size during validation and testing (<= 16 for GTX 1660 SUPER).
        num_workers: Number of dataloader worker processes.

    Returns:
        Folder datamodule ready for Anomalib Engine execution.
    """
    root_path = Path(root)
    if not root_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        if (project_root / root_path).exists():
            root_path = (project_root / root_path).resolve()
        else:
            root_path = root_path.resolve()

    # Normalize normal_test_dir in case "null", "none", or empty string is passed from YAML
    clean_normal_test_dir: Optional[str] = normal_test_dir
    if isinstance(clean_normal_test_dir, str) and clean_normal_test_dir.strip().lower() in ("", "null", "none"):
        clean_normal_test_dir = None

    # Check if normal_test_dir actually contains valid images; if not, fall back to split
    if clean_normal_test_dir is not None:
        candidate_test_path = root_path / clean_normal_test_dir
        has_images = candidate_test_path.is_dir() and any(
            p.suffix.lower() in IMAGE_EXTENSIONS for p in candidate_test_path.rglob("*") if p.is_file()
        )
        if not has_images:
            clean_normal_test_dir = None

    # Handle split ratio: when normal_test_dir is explicitly provided, no split should occur
    if clean_normal_test_dir is not None:
        clean_split_ratio = 0.0 if (normal_split_ratio is None or normal_split_ratio == 0) else float(normal_split_ratio)
    else:
        clean_split_ratio = 0.2 if normal_split_ratio is None else float(normal_split_ratio)

    # Construct dedicated preprocessing transform for datamodule (1:1 ROI + Native RGB)
    transform = Printer3DTransform(
        image_size=image_size,
        roi_crop=roi_crop,
    )

    # In Anomalib 2.6.2, 'image_size' is not an argument of Folder.__init__.
    # Custom resizing and normalization must be passed via augmentations.
    datamodule = Folder(
        name="printer3d",
        root=root_path,
        normal_dir=normal_dir,
        normal_test_dir=clean_normal_test_dir,
        abnormal_dir=abnormal_dir,
        mask_dir=mask_dir,
        normal_split_ratio=clean_split_ratio,
        augmentations=transform,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    return datamodule
