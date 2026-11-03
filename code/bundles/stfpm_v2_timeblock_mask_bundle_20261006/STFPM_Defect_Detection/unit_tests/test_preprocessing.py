"""Unit tests for the 3D printer preprocessing pipeline and predictor."""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import tv_tensors

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    DEFAULT_MIN_FILAMENT_RATIO,
    DEFAULT_ROI_CROP,
    Printer3DPreprocessor,
    Printer3DTransform,
    apply_spatial_patch_mask,
    check_filament_presence,
    crop_roi_numpy,
    crop_roi_tensor,
    validate_input_image,
)
from src.data.datamodule import build_datamodule


class TestGatekeeper(unittest.TestCase):
    """Tests for the input validation gatekeeper."""

    def test_none_input(self):
        is_valid, msg, arr = validate_input_image(None)
        self.assertFalse(is_valid)
        self.assertIn("None", msg)
        self.assertIsNone(arr)

    def test_nonexistent_file(self):
        is_valid, msg, arr = validate_input_image("non_existent_file_12345.jpg")
        self.assertFalse(is_valid)
        self.assertIn("does not exist", msg)
        self.assertIsNone(arr)

    def test_empty_array(self):
        empty_arr = np.array([], dtype=np.uint8)
        is_valid, msg, arr = validate_input_image(empty_arr)
        self.assertFalse(is_valid)
        self.assertIn("empty", msg)

    def test_pitch_black_array(self):
        black_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        is_valid, msg, arr = validate_input_image(black_arr)
        self.assertFalse(is_valid)
        self.assertIn("pitch black", msg)

    def test_tiny_image_below_min_dim(self):
        tiny_arr = np.ones((8, 8, 3), dtype=np.uint8) * 128
        is_valid, msg, arr = validate_input_image(tiny_arr, min_dimension=16)
        self.assertFalse(is_valid)
        self.assertIn("below minimum threshold", msg)

    def test_valid_numpy_array(self):
        valid_arr = np.random.randint(10, 240, (1200, 1600, 3), dtype=np.uint8)
        is_valid, msg, arr = validate_input_image(valid_arr)
        self.assertTrue(is_valid)
        self.assertIsNone(msg)
        self.assertIsNotNone(arr)

    def test_torch_tensor_input(self):
        tensor = torch.randint(10, 240, (3, 1200, 1600), dtype=torch.uint8)
        is_valid, msg, arr = validate_input_image(tensor)
        self.assertTrue(is_valid)
        self.assertIsNone(msg)
        self.assertEqual(arr.shape, (1200, 1600, 3))

    def test_nan_float_array(self):
        nan_arr = np.ones((100, 100, 3), dtype=np.float32)
        nan_arr[50, 50, 0] = np.nan
        is_valid, msg, arr = validate_input_image(nan_arr)
        self.assertFalse(is_valid)
        self.assertIn("NaN or Inf", msg)
        self.assertIsNone(arr)

    def test_inf_float_array(self):
        inf_arr = np.ones((100, 100, 3), dtype=np.float32)
        inf_arr[50, 50, 0] = np.inf
        is_valid, msg, arr = validate_input_image(inf_arr)
        self.assertFalse(is_valid)
        self.assertIn("NaN or Inf", msg)
        self.assertIsNone(arr)


class TestROICrop(unittest.TestCase):
    """Tests for 1:1 ROI cropping logic on numpy and tensors."""

    def test_crop_roi_numpy_standard_dimensions(self):
        # 1200x1600 image -> 1000x1000 with default 1:1 square ROI [0, 250, 1000, 1250]
        img = np.zeros((1200, 1600, 3), dtype=np.uint8)
        cropped = crop_roi_numpy(img, roi_crop=DEFAULT_ROI_CROP)
        self.assertEqual(cropped.shape, (1000, 1000, 3))

    def test_crop_roi_numpy_square_unchanged(self):
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        cropped = crop_roi_numpy(img, roi_crop=DEFAULT_ROI_CROP)
        self.assertEqual(cropped.shape, (512, 512, 3))

    def test_crop_roi_tensor_standard_dimensions(self):
        tensor = torch.zeros((3, 1200, 1600), dtype=torch.float32)
        cropped = crop_roi_tensor(tensor, roi_crop=DEFAULT_ROI_CROP)
        self.assertEqual(cropped.shape, (3, 1000, 1000))

    def test_crop_roi_tensor_fallback_center_crop(self):
        # Non-standard non-square image
        tensor = torch.zeros((3, 600, 800), dtype=torch.float32)
        cropped = crop_roi_tensor(tensor, roi_crop=DEFAULT_ROI_CROP)
        # Should fallback to center 600x600 crop
        self.assertEqual(cropped.shape, (3, 600, 600))


class TestFilamentPresence(unittest.TestCase):
    """Tests for Otsu 5% filament presence validation."""

    def test_filament_presence_positive(self):
        # 100x100 image with 30% bright filament (gray ~ 200) and 70% dark bed (gray ~ 30)
        img = np.full((100, 100, 3), 30, dtype=np.uint8)
        img[:30, :] = 200  # 30% area
        is_present, ratio, mask = check_filament_presence(img, min_ratio=0.05)
        self.assertTrue(is_present)
        self.assertAlmostEqual(ratio, 0.30, delta=0.02)
        self.assertEqual(mask.shape, (100, 100))

    def test_filament_presence_negative_empty_bed(self):
        # 100x100 image with only 2% bright area (below 5% threshold)
        img = np.full((100, 100, 3), 30, dtype=np.uint8)
        img[:2, :] = 200  # 2% area
        is_present, ratio, mask = check_filament_presence(img, min_ratio=0.05)
        self.assertFalse(is_present)
        self.assertLess(ratio, 0.05)

    def test_gatekeeper_rejects_empty_bed(self):
        # Image with 2% filament checked with min_filament_ratio=0.05
        img = np.full((100, 100, 3), 30, dtype=np.uint8)
        img[:2, :] = 200
        is_valid, msg, arr = validate_input_image(img, min_filament_ratio=0.05)
        self.assertFalse(is_valid)
        self.assertIn("Empty bed detected", msg)

    def test_gatekeeper_accepts_valid_filament(self):
        # Image with 20% filament
        img = np.full((100, 100, 3), 30, dtype=np.uint8)
        img[:20, :] = 200
        is_valid, msg, arr = validate_input_image(img, min_filament_ratio=0.05)
        self.assertTrue(is_valid)
        self.assertIsNone(msg)

    def test_gatekeeper_roi_crop_filament_check(self):
        # 200x200 image where only ROI [50, 50, 150, 150] (100x100) has filament
        img = np.full((200, 200, 3), 30, dtype=np.uint8)
        img[50:100, 50:150] = 200
        # Global ratio = 5000 / 40000 = 12.5%
        # ROI ratio = 5000 / 10000 = 50%
        is_valid_no_roi, _, _ = validate_input_image(img, min_filament_ratio=0.20)
        self.assertFalse(is_valid_no_roi)

        is_valid_roi, _, _ = validate_input_image(img, min_filament_ratio=0.20, roi_crop=[50, 50, 150, 150])
        self.assertTrue(is_valid_roi)



class TestSpatialPatchMask(unittest.TestCase):
    """Tests for spatial boundary patch masking."""

    def test_border_masking(self):
        # 100x100 anomaly map filled with ones
        anomaly_map = torch.ones((1, 1, 100, 100), dtype=torch.float32)
        # 10% margin -> outer 10 pixels should be 0
        masked = apply_spatial_patch_mask(anomaly_map, margin_ratio=0.10)
        self.assertEqual(masked[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(masked[0, 0, 9, 9].item(), 0.0)
        self.assertEqual(masked[0, 0, 50, 50].item(), 1.0)

    def test_border_masking_half_margin_returns_zero(self):
        anomaly_map = torch.ones((1, 1, 100, 100), dtype=torch.float32)
        masked = apply_spatial_patch_mask(anomaly_map, margin_ratio=0.50)
        self.assertEqual(masked.sum().item(), 0.0)

    def test_border_masking_patch_grid(self):
        # 32x32 patch score grid with 10% margin
        patch_grid = torch.ones((2, 1, 32, 32), dtype=torch.float32)
        masked = apply_spatial_patch_mask(patch_grid, margin_ratio=0.10)
        self.assertEqual(masked.shape, (2, 1, 32, 32))
        # Margin: int(32 * 0.1) = 3 patches on each side
        self.assertEqual(masked[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(masked[0, 0, 2, 2].item(), 0.0)
        self.assertEqual(masked[0, 0, 16, 16].item(), 1.0)


class TestPrinter3DTransform(unittest.TestCase):
    """Tests for torchvision v2 Transform implementation."""

    def setUp(self):
        self.transform = Printer3DTransform(
            image_size=(256, 256),
            roi_crop=(0, 250, 1000, 1250),
        )

    def test_image_transform(self):
        raw_img = tv_tensors.Image(torch.rand(3, 1200, 1600, dtype=torch.float32))
        out_img = self.transform(raw_img)
        self.assertIsInstance(out_img, tv_tensors.Image)
        self.assertEqual(out_img.shape, torch.Size([3, 256, 256]))
        self.assertEqual(out_img.dtype, torch.float32)

    def test_batched_image_transform(self):
        raw_batch = torch.rand(4, 3, 1200, 1600, dtype=torch.float32)
        out_batch = self.transform(raw_batch)
        self.assertEqual(out_batch.shape, torch.Size([4, 3, 256, 256]))
        self.assertEqual(out_batch.dtype, torch.float32)

    def test_mask_transform(self):
        raw_mask = tv_tensors.Mask(torch.zeros((1200, 1600), dtype=torch.uint8))
        out_mask = self.transform(raw_mask)
        self.assertIsInstance(out_mask, tv_tensors.Mask)
        self.assertEqual(out_mask.shape, torch.Size([256, 256]))
        self.assertEqual(out_mask.dtype, torch.uint8)

    def test_image_and_mask_transform_pair(self):
        raw_img = tv_tensors.Image(torch.rand(3, 1200, 1600, dtype=torch.float32))
        raw_mask = tv_tensors.Mask(torch.zeros((1200, 1600), dtype=torch.uint8))
        out_img, out_mask = self.transform(raw_img, raw_mask)
        self.assertEqual(out_img.shape, torch.Size([3, 256, 256]))
        self.assertEqual(out_mask.shape, torch.Size([256, 256]))


    def test_2d_tensor_transform(self):
        raw_2d = torch.rand(1200, 1600, dtype=torch.float32)
        out = self.transform(raw_2d)
        self.assertEqual(out.shape, torch.Size([3, 256, 256]))
        self.assertEqual(out.dtype, torch.float32)


class TestPrinter3DPreprocessor(unittest.TestCase):
    """Tests for standalone preprocessor."""

    def test_preprocess_numpy_bgr(self):
        preprocessor = Printer3DPreprocessor(device="cpu")
        sample_bgr = np.random.randint(10, 240, (1200, 1600, 3), dtype=np.uint8)
        tensor = preprocessor.preprocess_image(sample_bgr, is_bgr=True)
        self.assertEqual(tensor.shape, torch.Size([1, 3, 256, 256]))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))

    def test_preprocess_float64_numpy(self):
        preprocessor = Printer3DPreprocessor(device="cpu")
        sample_float64 = np.random.rand(1200, 1600, 3)  # float64
        tensor = preprocessor.preprocess_image(sample_float64, is_bgr=True)
        self.assertEqual(tensor.shape, torch.Size([1, 3, 256, 256]))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_zero_interpolation_drift(self):
        # Verify preprocessor produces identical output to Printer3DTransform (zero drift)
        preprocessor = Printer3DPreprocessor(device="cpu")
        sample_bgr = np.random.randint(10, 240, (1200, 1600, 3), dtype=np.uint8)
        arr_rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)
        tensor_raw = torch.from_numpy(arr_rgb).permute(2, 0, 1).float() / 255.0
        expected = preprocessor.get_torchvision_transform()(tv_tensors.Image(tensor_raw)).unsqueeze(0)
        actual = preprocessor.preprocess_image(sample_bgr, is_bgr=True)
        torch.testing.assert_close(actual, expected)


class TestDataModuleIntegration(unittest.TestCase):
    """Tests for Anomalib DataModule construction with updated pipeline."""

    def test_build_datamodule_execution(self):
        # Verify Folder initialization does not raise TypeError for image_size
        dm = build_datamodule(
            root="data/printer3d",
            normal_dir="train/good",
            normal_test_dir=None,
            abnormal_dir="test/defect",
            normal_split_ratio=0.2,
            image_size=(256, 256),
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
        )
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        self.assertTrue(hasattr(batch, "image"))
        self.assertEqual(batch.image.shape, torch.Size([4, 3, 256, 256]))

    def test_build_datamodule_with_explicit_normal_test_dir(self):
        # Verify Folder initialization with explicit test/good normal directory preserving all 1600 train images
        dm = build_datamodule(
            root="data/printer3d",
            normal_dir="train/good",
            normal_test_dir="test/good",
            abnormal_dir="test/defect",
            normal_split_ratio=None,
            image_size=(256, 256),
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
        )
        dm.setup()
        self.assertEqual(len(dm.train_data), 1600)
        self.assertEqual(len(dm.test_data), 1200)

    def test_compute_unsupervised_threshold_mock(self):
        from unittest.mock import MagicMock
        from src.train import compute_unsupervised_threshold
        mock_model = MagicMock()
        mock_model.model.memory_bank = torch.randn(50, 32)
        mock_model.model.nearest_neighbors.return_value = (
            torch.stack([torch.zeros(50), torch.ones(50) * 4.2], dim=1),
            None,
        )
        res = compute_unsupervised_threshold(mock_model, method="percentile")
        self.assertAlmostEqual(res["threshold"], 4.2, places=4)
        self.assertEqual(res["method"], "unsupervised_percentile")

    def test_compute_unsupervised_threshold_mu_3sigma(self):
        from unittest.mock import MagicMock
        from src.train import compute_unsupervised_threshold
        mock_model = MagicMock()
        mock_model.model.memory_bank = torch.randn(10, 16)
        # Distances: constant 2.0 (mean=2.0, std=0.0 -> mu + 3*sigma = 2.0)
        mock_model.model.nearest_neighbors.return_value = (
            torch.stack([torch.zeros(10), torch.ones(10) * 2.0], dim=1),
            None,
        )
        res = compute_unsupervised_threshold(mock_model, method="mu_3sigma")
        self.assertAlmostEqual(res["threshold"], 2.0, places=4)
        self.assertEqual(res["method"], "unsupervised_mu_3sigma")
        self.assertAlmostEqual(res["mu"], 2.0, places=4)
        self.assertAlmostEqual(res["sigma"], 0.0, places=4)

    def test_compute_unsupervised_threshold_real_postprocessor(self):
        """Verifies compute_unsupervised_threshold synchronizes to real PostProcessor without setter AttributeError."""
        from unittest.mock import MagicMock
        from anomalib.post_processing import PostProcessor
        from src.train import compute_unsupervised_threshold

        mock_model = MagicMock()
        mock_model.post_processor = PostProcessor()
        mock_model.model.memory_bank = torch.randn(20, 16)
        mock_model.model.nearest_neighbors.return_value = (
            torch.stack([torch.zeros(20), torch.ones(20) * 5.5], dim=1),
            None,
        )
        res = compute_unsupervised_threshold(mock_model, method="percentile")
        self.assertAlmostEqual(res["threshold"], 5.5, places=4)
        self.assertAlmostEqual(mock_model.post_processor.image_threshold.item(), 5.5, places=4)

    def test_compute_unsupervised_threshold_fallback_empty(self):
        from unittest.mock import MagicMock
        from src.train import compute_unsupervised_threshold

        mock_model = MagicMock()
        mock_model.model.memory_bank = None
        mock_model.post_processor.image_threshold = torch.tensor(float("nan"))
        res = compute_unsupervised_threshold(mock_model, fallback_threshold=0.75)
        self.assertEqual(res["threshold"], 0.5)
        self.assertEqual(res["method"], "fallback_model")


class TestPredictorBatchAndFailSafe(unittest.TestCase):
    """Tests for predictor batch processing and fail-safe handling."""

    @classmethod
    def setUpClass(cls):
        ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "stfpm_model.ckpt"
        if not ckpt_path.exists():
            ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "patchcore_model.ckpt"
        if ckpt_path.exists():
            from src.models.predictor import Printer3DDefectPredictor
            cls.predictor = Printer3DDefectPredictor(
                checkpoint_path=ckpt_path,
                device="cpu",
            )
        else:
            cls.predictor = None

    def test_predict_batch_fail_safe_mixed(self):
        if self.predictor is None:
            self.skipTest("Checkpoint not available")

        valid_arr = np.random.randint(10, 240, (1200, 1600, 3), dtype=np.uint8)
        inputs = [valid_arr, None, valid_arr, "non_existent_file.jpg"]
        results = self.predictor.predict_batch(inputs, fail_safe=True)

        self.assertEqual(len(results), 4)
        self.assertIn(results[0].decision, ("GOOD", "DEFECT"))
        self.assertEqual(results[1].decision, "ERROR")
        self.assertIn(results[2].decision, ("GOOD", "DEFECT"))
        self.assertEqual(results[3].decision, "ERROR")

    def test_predict_single_matches_batch(self):
        if self.predictor is None:
            self.skipTest("Checkpoint not available")

        valid_arr = np.random.randint(10, 240, (1200, 1600, 3), dtype=np.uint8)
        single_res = self.predictor.predict(valid_arr)
        batch_res = self.predictor.predict_batch([valid_arr])[0]

        self.assertEqual(single_res.decision, batch_res.decision)
        self.assertAlmostEqual(single_res.anomaly_score, batch_res.anomaly_score, places=4)

    def test_spatial_mask_margin_precedence(self):
        if self.predictor is None:
            self.skipTest("Checkpoint not available")

        from src.models.predictor import Printer3DDefectPredictor
        ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "stfpm_model.ckpt"
        if not ckpt_path.exists():
            ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "patchcore_model.ckpt"
        pred = Printer3DDefectPredictor(
            checkpoint_path=ckpt_path,
            device="cpu",
            spatial_mask_margin=0.08,
        )
        self.assertEqual(pred.spatial_mask_margin, 0.08)

    def test_torch_tensor_prediction(self):
        if self.predictor is None:
            self.skipTest("Checkpoint not available")

        tensor_img = torch.randint(10, 240, (3, 1200, 1600), dtype=torch.uint8)
        res = self.predictor.predict(tensor_img)
        self.assertIn(res.decision, ("GOOD", "DEFECT"))
        self.assertGreater(res.anomaly_score, 0.0)

    def test_predict_batch_large_chunking(self):
        """Verifies predict_batch chunks lists larger than 16 (hardware limit) safely."""
        if self.predictor is None:
            self.skipTest("Checkpoint not available")

        valid_arr = np.random.randint(10, 240, (1200, 1600, 3), dtype=np.uint8)
        # 25 inputs: mixed valid and invalid crossing the 16-item boundary
        inputs = [valid_arr if i not in (5, 18) else None for i in range(25)]
        results = self.predictor.predict_batch(inputs, fail_safe=True, batch_size=10)

        self.assertEqual(len(results), 25)
        for i, res in enumerate(results):
            if i in (5, 18):
                self.assertEqual(res.decision, "ERROR")
            else:
                self.assertIn(res.decision, ("GOOD", "DEFECT"))
                self.assertGreater(res.anomaly_score, 0.0)


if __name__ == "__main__":
    unittest.main()
