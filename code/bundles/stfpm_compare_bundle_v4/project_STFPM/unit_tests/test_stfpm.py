"""Unit tests for STFPM architecture: builder, forward inference, spatial masking, and predictor."""

import json
import shutil
import tempfile
import unittest
import sys
from pathlib import Path
from typing import Sequence

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
from anomalib.models import Stfpm

from src.data.preprocessing import apply_spatial_patch_mask
from src.models import PredictionResult, Printer3DDefectPredictor, build_stfpm_model



class TestSTFPMBuilder(unittest.TestCase):
    """Tests for build_stfpm_model and model configuration."""

    def test_build_default_resnet18(self):
        """Verifies default STFPM model construction with ResNet-18 backbone."""
        model = build_stfpm_model()
        self.assertIsInstance(model, Stfpm)
        self.assertEqual(list(model.model.teacher_model.layers), ["layer1", "layer2", "layer3"])
        self.assertEqual(model.model.backbone, "resnet18")
        self.assertIsNone(model.pre_processor)
        self.assertIsNone(model.post_processor)

    def test_build_resnet34(self):
        """Verifies STFPM model construction with ResNet-34 backbone."""
        model = build_stfpm_model(backbone="resnet34")
        self.assertIsInstance(model, Stfpm)
        self.assertEqual(model.model.backbone, "resnet34")
        self.assertEqual(list(model.model.teacher_model.layers), ["layer1", "layer2", "layer3"])

    def test_build_custom_layers(self):
        """Verifies STFPM model construction with custom layers."""
        model = build_stfpm_model(backbone="resnet18", layers=["layer1", "layer2"])
        self.assertEqual(list(model.model.teacher_model.layers), ["layer1", "layer2"])


    def test_build_none_layers_defaults_properly(self):
        """Passing layers=None should safely fallback to default 3 layers."""
        model = build_stfpm_model(layers=None)
        self.assertEqual(list(model.model.teacher_model.layers), ["layer1", "layer2", "layer3"])

    def test_build_tuple_layers(self):
        """Passing layers as tuple should be accepted and converted to list."""
        model = build_stfpm_model(layers=("layer1", "layer2"))
        self.assertEqual(list(model.model.teacher_model.layers), ["layer1", "layer2"])

    def test_build_custom_optimizer_hyperparams(self):
        """Verifies custom learning rate, momentum, and weight decay are applied."""
        model = build_stfpm_model(
            backbone="resnet18",
            learning_rate=0.2,
            momentum=0.85,
            weight_decay=0.0005,
        )
        opt = model.configure_optimizers()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.2)
        self.assertAlmostEqual(opt.param_groups[0]["momentum"], 0.85)
        self.assertAlmostEqual(opt.param_groups[0]["weight_decay"], 0.0005)

    def test_exports_in_src_models(self):
        """Verifies expected public symbols are exposed from src.models."""
        from src.models import PredictionResult, Printer3DDefectPredictor, build_stfpm_model
        self.assertTrue(callable(build_stfpm_model))
        self.assertTrue(callable(Printer3DDefectPredictor))
        self.assertTrue(callable(PredictionResult))


class TestSTFPMForwardInference(unittest.TestCase):
    """Tests for STFPM forward tensor shapes and anomaly outputs."""

    @classmethod
    def setUpClass(cls):
        cls.model = build_stfpm_model(backbone="resnet18")
        cls.model.eval()

    @torch.no_grad()
    def test_forward_single_tensor_shape(self):
        """Verifies forward inference shapes on a single image input (B=1, 256x256)."""
        x = torch.randn(1, 3, 256, 256)
        out = self.model.model(x)
        self.assertTrue(hasattr(out, "pred_score"))
        self.assertTrue(hasattr(out, "anomaly_map"))
        self.assertEqual(out.pred_score.shape, torch.Size([1, 1]))
        self.assertEqual(out.anomaly_map.shape, torch.Size([1, 1, 256, 256]))

    @torch.no_grad()
    def test_forward_single_tensor_512_resolution(self):
        """Verifies forward inference shapes on 512x512 resolution (project default)."""
        x = torch.randn(1, 3, 512, 512)
        out = self.model.model(x)
        self.assertEqual(out.pred_score.shape, torch.Size([1, 1]))
        self.assertEqual(out.anomaly_map.shape, torch.Size([1, 1, 512, 512]))

    @torch.no_grad()
    def test_forward_batch_tensor_shape(self):
        """Verifies forward inference shapes on a multi-image batch (B=4)."""
        x = torch.randn(4, 3, 256, 256)
        out = self.model.model(x)
        self.assertEqual(out.pred_score.shape, torch.Size([4, 1]))
        self.assertEqual(out.anomaly_map.shape, torch.Size([4, 1, 256, 256]))

    @torch.no_grad()
    def test_forward_scores_validity(self):
        """Verifies anomaly scores and maps are finite non-negative numbers."""
        x = torch.randn(2, 3, 256, 256)
        out = self.model.model(x)
        self.assertTrue(torch.isfinite(out.pred_score).all())
        self.assertTrue(torch.isfinite(out.anomaly_map).all())
        self.assertTrue((out.pred_score >= 0.0).all())



class TestSpatialMarginMask(unittest.TestCase):
    """Tests for bed boundary suppression via spatial patch masking."""

    def test_margin_zero_returns_identical(self):
        """Margin 0.0 should leave the anomaly map unchanged."""
        anomaly_map = torch.ones(2, 1, 256, 256)
        masked = apply_spatial_patch_mask(anomaly_map, margin_ratio=0.0)
        self.assertTrue(torch.equal(anomaly_map, masked))

    def test_margin_suppresses_boundary_artifacts(self):
        """Boundary margin should zero out edge values while preserving center."""
        anomaly_map = torch.zeros(1, 1, 256, 256)
        # Put high artifact strictly on the border (e.g. y=2, x=2)
        anomaly_map[0, 0, 2, 2] = 10.0
        # Put genuine defect near the center (y=128, x=128)
        anomaly_map[0, 0, 128, 128] = 5.0

        # Without mask, max is 10.0
        self.assertAlmostEqual(torch.amax(anomaly_map).item(), 10.0)

        # With 5% margin, border is suppressed (256 * 0.05 = 12 pixels border)
        masked = apply_spatial_patch_mask(anomaly_map, margin_ratio=0.05)
        # Border artifact (2, 2) should be zeroed
        self.assertEqual(masked[0, 0, 2, 2].item(), 0.0)
        # Center defect (128, 128) should be intact
        self.assertEqual(masked[0, 0, 128, 128].item(), 5.0)
        # Max score is now the center defect score
        self.assertAlmostEqual(torch.amax(masked).item(), 5.0)

    def test_margin_half_or_greater_zeros_all(self):
        """Margin ratio >= 0.5 masks the entire map."""
        anomaly_map = torch.ones(1, 1, 100, 100)
        masked = apply_spatial_patch_mask(anomaly_map, margin_ratio=0.5)
        self.assertTrue((masked == 0.0).all())


class TestPrinter3DDefectPredictor(unittest.TestCase):
    """Tests for Printer3DDefectPredictor inference, batching, and fail-safe handling."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="test_stfpm_predictor_")
        cls.tmp_path = Path(cls.temp_dir)

        # Build and save a minimal STFPM checkpoint
        model = build_stfpm_model(backbone="resnet18")
        trainer = pl.Trainer(default_root_dir=cls.temp_dir, max_epochs=1, barebones=True)
        trainer.strategy.connect(model)
        cls.ckpt_path = cls.tmp_path / "stfpm_model.ckpt"
        trainer.save_checkpoint(cls.ckpt_path)

        # Create model metadata
        cls.metadata_path = cls.tmp_path / "model_metadata.json"
        metadata = {
            "threshold": 0.25,
            "image_threshold": 0.25,
            "image_size": [256, 256],
            "roi_crop": [0, 200, 1050, 1400],
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": [8, 8],
            "spatial_mask_margin": 0.05,
        }
        with open(cls.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        # Create sample test images
        cls.valid_img_path = cls.tmp_path / "sample_valid.jpg"
        dummy_img = np.full((1200, 1600, 3), 128, dtype=np.uint8)
        # Draw some features so it's not a degenerate blank image
        cv2.circle(dummy_img, (800, 600), 100, (200, 200, 200), -1)
        cv2.imwrite(str(cls.valid_img_path), dummy_img)

        cls.corrupted_img_path = cls.tmp_path / "corrupted.jpg"
        with open(cls.corrupted_img_path, "wb") as f:
            f.write(b"not a valid image format junk bytes")

        cls.empty_img_path = cls.tmp_path / "empty.jpg"
        cls.empty_img_path.touch()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_init_missing_checkpoint_raises(self):
        """Non-existent checkpoint file should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            Printer3DDefectPredictor(checkpoint_path=self.tmp_path / "non_existent.ckpt")

    def test_init_loads_metadata_correctly(self):
        """Predictor should properly load threshold and configuration from metadata."""
        predictor = Printer3DDefectPredictor(
            checkpoint_path=self.ckpt_path,
            metadata_path=self.metadata_path,
        )
        self.assertAlmostEqual(predictor.threshold, 0.25)
        self.assertEqual(predictor.image_size, (256, 256))
        self.assertEqual(predictor.roi_crop, (0, 200, 1050, 1400))
        self.assertAlmostEqual(predictor.spatial_mask_margin, 0.05)

    def test_forward_model_tensor_shapes(self):
        """_forward_model should return 1D tensor of shape (B,)."""
        predictor = Printer3DDefectPredictor(checkpoint_path=self.ckpt_path)
        batch = torch.randn(3, 3, 256, 256)
        scores = predictor._forward_model(batch, margin=0.0)
        self.assertEqual(scores.shape, torch.Size([3]))
        self.assertTrue(torch.isfinite(scores).all())

    def test_forward_model_with_margin(self):
        """_forward_model with margin > 0 applies spatial masking properly."""
        predictor = Printer3DDefectPredictor(checkpoint_path=self.ckpt_path)
        batch = torch.randn(2, 3, 256, 256)
        scores_masked = predictor._forward_model(batch, margin=0.05)
        self.assertEqual(scores_masked.shape, torch.Size([2]))
        self.assertTrue(torch.isfinite(scores_masked).all())

    def test_predict_single_image_path(self):
        """predict() handles Path and string path input."""
        predictor = Printer3DDefectPredictor(checkpoint_path=self.ckpt_path)
        res = predictor.predict(self.valid_img_path)
        self.assertIsInstance(res, PredictionResult)
        self.assertIn(res.decision, ["GOOD", "DEFECT"])
        self.assertEqual(res.is_anomaly, res.anomaly_score >= res.threshold)
        self.assertGreater(res.anomaly_score, 0.0)

        # String path test
        res_str = predictor.predict(str(self.valid_img_path))
        self.assertIsInstance(res_str, PredictionResult)
        self.assertIn(res_str.decision, ["GOOD", "DEFECT"])

    def test_predict_single_image_ndarray(self):
        """predict() handles OpenCV BGR ndarray input."""
        predictor = Printer3DDefectPredictor(checkpoint_path=self.ckpt_path)
        arr = np.full((300, 300, 3), 100, dtype=np.uint8)
        cv2.rectangle(arr, (50, 50), (200, 200), (180, 180, 180), -1)
        res = predictor.predict(arr)
        self.assertIsInstance(res, PredictionResult)
        self.assertIn(res.decision, ["GOOD", "DEFECT"])
        self.assertIsNone(res.image_path)

    def test_predict_custom_threshold_override(self):
        """Custom threshold override should change decision accordingly."""
        predictor = Printer3DDefectPredictor(checkpoint_path=self.ckpt_path)
        # Extremely high threshold -> always GOOD
        res_high = predictor.predict(self.valid_img_path, custom_threshold=999.0)
        self.assertEqual(res_high.decision, "GOOD")
        self.assertFalse(res_high.is_anomaly)
        self.assertEqual(res_high.threshold, 999.0)

        # Extremely low threshold -> always DEFECT
        res_low = predictor.predict(self.valid_img_path, custom_threshold=0.00001)
        self.assertEqual(res_low.decision, "DEFECT")
        self.assertTrue(res_low.is_anomaly)
        self.assertEqual(res_low.threshold, 0.00001)

    def test_predict_batch_mixed_inputs(self):
        """predict_batch handles mixed inputs and enforces hardware-constrained batch size."""
        predictor = Printer3DDefectPredictor(checkpoint_path=self.ckpt_path)
        arr = np.full((400, 400, 3), 150, dtype=np.uint8)
        inputs = [self.valid_img_path, arr, str(self.valid_img_path)]
        results = predictor.predict_batch(inputs, batch_size=2)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn(r.decision, ["GOOD", "DEFECT"])

    def test_predict_batch_empty_input(self):
        """predict_batch([]) returns empty list."""
        predictor = Printer3DDefectPredictor(checkpoint_path=self.ckpt_path)
        self.assertEqual(predictor.predict_batch([]), [])

    def test_fail_safe_on_invalid_inputs(self):
        """Invalid inputs return ERROR PredictionResult when fail_safe=True."""
        predictor = Printer3DDefectPredictor(checkpoint_path=self.ckpt_path)

        # 1. Non-existent file
        res_missing = predictor.predict(self.tmp_path / "does_not_exist.jpg", fail_safe=True)
        self.assertEqual(res_missing.decision, "ERROR")
        self.assertIn("does not exist", res_missing.error_message)

        # 2. Corrupted file
        res_corrupt = predictor.predict(self.corrupted_img_path, fail_safe=True)
        self.assertEqual(res_corrupt.decision, "ERROR")
        self.assertIn("Failed to decode", res_corrupt.error_message)

        # 3. Empty file
        res_empty = predictor.predict(self.empty_img_path, fail_safe=True)
        self.assertEqual(res_empty.decision, "ERROR")
        self.assertIn("empty", res_empty.error_message)

        # 4. None input
        res_none = predictor.predict(None, fail_safe=True)
        self.assertEqual(res_none.decision, "ERROR")
        self.assertIn("missing input", res_none.error_message)

        # 5. Pitch black array
        black_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        res_black = predictor.predict(black_arr, fail_safe=True)
        self.assertEqual(res_black.decision, "ERROR")
        self.assertIn("pitch black", res_black.error_message)

    def test_forward_model_enforces_eval_mode(self):
        """_forward_model should automatically set model to eval mode if set to train."""
        predictor = Printer3DDefectPredictor(checkpoint_path=self.ckpt_path)
        predictor.model.train(True)
        self.assertTrue(predictor.model.training)
        batch = torch.randn(1, 3, 256, 256)
        scores = predictor._forward_model(batch)
        self.assertFalse(predictor.model.training)
        self.assertEqual(scores.shape, torch.Size([1]))

    def test_predict_single_image_grayscale_2d(self):
        """predict() handles 2D grayscale numpy array input (H, W)."""
        predictor = Printer3DDefectPredictor(checkpoint_path=self.ckpt_path)
        gray_arr = np.full((300, 300), 120, dtype=np.uint8)
        cv2.circle(gray_arr, (150, 150), 50, 220, -1)
        res = predictor.predict(gray_arr)
        self.assertIsInstance(res, PredictionResult)
        self.assertIn(res.decision, ["GOOD", "DEFECT"])
        self.assertIsNone(res.image_path)

    def test_predict_single_image_512_resolution(self):
        """predict() handles 512x512 image resolution cleanly."""
        predictor = Printer3DDefectPredictor(checkpoint_path=self.ckpt_path)
        img_512 = np.full((512, 512, 3), 130, dtype=np.uint8)
        res = predictor.predict(img_512)
        self.assertIsInstance(res, PredictionResult)
        self.assertIn(res.decision, ["GOOD", "DEFECT"])

    def test_predict_batch_generator_iterable(self):
        """predict_batch handles general iterables (generators, iterators)."""
        predictor = Printer3DDefectPredictor(checkpoint_path=self.ckpt_path)
        gen = (x for x in [self.valid_img_path, str(self.valid_img_path)])
        results = predictor.predict_batch(gen, batch_size=2)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn(r.decision, ["GOOD", "DEFECT"])

    def test_init_loads_model_image_threshold_fallback(self):
        """When metadata is missing, threshold falls back to model buffer if present."""
        # Create an isolated subdirectory so candidate_meta does not find model_metadata.json
        isolated_dir = self.tmp_path / "isolated_no_meta"
        isolated_dir.mkdir(exist_ok=True)
        ckpt_no_meta = isolated_dir / "model_no_meta.ckpt"

        model = Stfpm(backbone="resnet18")
        model.post_processor._image_threshold.copy_(torch.tensor(0.42))
        trainer = pl.Trainer(default_root_dir=str(isolated_dir), barebones=True)
        trainer.strategy.connect(model)
        trainer.save_checkpoint(ckpt_no_meta)

        predictor = Printer3DDefectPredictor(checkpoint_path=ckpt_no_meta)
        self.assertAlmostEqual(predictor.threshold, 0.42, places=4)


    def test_fail_safe_on_nan_and_small_inputs(self):
        """Arrays with NaNs or sub-minimum dimensions return ERROR when fail_safe=True."""
        predictor = Printer3DDefectPredictor(checkpoint_path=self.ckpt_path)

        # 1. NaN array
        nan_arr = np.full((100, 100, 3), np.nan, dtype=np.float32)
        res_nan = predictor.predict(nan_arr, fail_safe=True)
        self.assertEqual(res_nan.decision, "ERROR")
        self.assertIn("NaN", res_nan.error_message)

        # 2. Sub-minimum dimensions (8x8)
        small_arr = np.full((8, 8, 3), 128, dtype=np.uint8)
        res_small = predictor.predict(small_arr, fail_safe=True)
        self.assertEqual(res_small.decision, "ERROR")
        self.assertIn("below minimum threshold", res_small.error_message)

    def test_fail_safe_false_raises_exception(self):
        """When fail_safe=False, invalid inputs raise ValueError."""
        predictor = Printer3DDefectPredictor(checkpoint_path=self.ckpt_path)
        with self.assertRaises(ValueError):
            predictor.predict(self.tmp_path / "non_existent.jpg", fail_safe=False)

    def test_prediction_result_string_format(self):
        """PredictionResult string conversion produces readable reports."""
        res_good = PredictionResult("test.jpg", is_anomaly=False, anomaly_score=0.12, threshold=0.3, decision="GOOD")
        self.assertIn("GOOD", str(res_good))
        self.assertIn("0.1200", str(res_good))

        res_defect = PredictionResult("test.jpg", is_anomaly=True, anomaly_score=0.55, threshold=0.3, decision="DEFECT")
        self.assertIn("DEFECT", str(res_defect))
        self.assertIn("0.5500", str(res_defect))

        res_error = PredictionResult("test.jpg", is_anomaly=False, anomaly_score=0.0, threshold=0.3, decision="ERROR", error_message="Corrupted image")
        self.assertIn("ERROR", str(res_error))
        self.assertIn("Corrupted image", str(res_error))

    def test_predict_cli_integration(self):
        """Root entry point predict.py works with STFPM checkpoint via CLI."""
        import subprocess
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--checkpoint",
            str(self.ckpt_path),
            "--image",
            str(self.valid_img_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        self.assertEqual(res.returncode, 0, f"predict.py failed: {res.stderr}")
        self.assertIn("3D 프린터 출력물 결함 판별기", res.stdout)
        self.assertIn("최종 판별 통계 요약", res.stdout)

    def test_predict_cli_integration_input_dir(self):
        """Root entry point predict.py works with --input-dir via CLI."""
        import subprocess
        # Create a directory with 2 test images
        img_dir = self.tmp_path / "cli_test_dir"
        img_dir.mkdir(exist_ok=True)
        img1 = np.full((200, 200, 3), 120, dtype=np.uint8)
        img2 = np.full((200, 200, 3), 140, dtype=np.uint8)
        cv2.imwrite(str(img_dir / "img1.png"), img1)
        cv2.imwrite(str(img_dir / "img2.png"), img2)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--checkpoint",
            str(self.ckpt_path),
            "--input-dir",
            str(img_dir),
            "--batch-size",
            "2",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        self.assertEqual(res.returncode, 0, f"predict.py --input-dir failed: {res.stderr}")
        self.assertIn("3D 프린터 출력물 결함 판별기", res.stdout)
        self.assertIn("총 2개 이미지 감지됨", res.stdout)
        self.assertIn("최종 판별 통계 요약", res.stdout)
        self.assertIn("전체 검사 대상: 2장", res.stdout)


class TestSTFPMTrainingAndThreshold(unittest.TestCase):
    """Tests for STFPM threshold calculation with dataloader and spatial margin masking."""

    def test_compute_unsupervised_threshold_with_dataloader(self):
        from unittest.mock import MagicMock
        from src.train import compute_unsupervised_threshold
        from anomalib.data.dataclasses.torch.base import InferenceBatch

        mock_model = MagicMock()
        mock_model.model.eval.return_value = None

        dummy_dataloader = [
            {"image": torch.randn(4, 3, 64, 64)},
            {"image": torch.randn(4, 3, 64, 64)},
        ]

        def mock_forward(imgs):
            b = imgs.shape[0]
            scores = torch.linspace(0.1, 0.2, b).unsqueeze(-1)
            maps = torch.zeros(b, 1, 64, 64)
            return InferenceBatch(pred_score=scores, pred_mask=None, anomaly_map=maps, pred_label=None)

        mock_model.model.side_effect = mock_forward

        res = compute_unsupervised_threshold(
            model=mock_model,
            dataloader=dummy_dataloader,
            device=torch.device("cpu"),
            method="percentile",
            percentile=0.99,
        )
        self.assertEqual(res["sample_count"], 8)
        self.assertIn("threshold", res)
        self.assertEqual(res["method"], "stfpm_unsupervised_percentile")
        self.assertAlmostEqual(res["threshold"], 0.2, places=2)

    def test_compute_unsupervised_threshold_with_spatial_margin(self):
        from unittest.mock import MagicMock
        from src.train import compute_unsupervised_threshold
        from anomalib.data.dataclasses.torch.base import InferenceBatch

        mock_model = MagicMock()
        mock_model.model.eval.return_value = None

        dummy_dataloader = [{"image": torch.randn(2, 3, 64, 64)}]

        def mock_forward(imgs):
            b = imgs.shape[0]
            maps = torch.zeros(b, 1, 64, 64)
            # Edge has 1.0, interior has 0.05
            maps[:, :, 0:6, :] = 1.0
            maps[:, :, :, 0:6] = 1.0
            maps[:, :, 10:54, 10:54] = 0.05
            scores = torch.tensor([[1.0], [1.0]])
            return InferenceBatch(pred_score=scores, pred_mask=None, anomaly_map=maps, pred_label=None)

        mock_model.model.side_effect = mock_forward

        # With margin=0.15 (15% margin), border artifacts (1.0) are suppressed
        res = compute_unsupervised_threshold(
            model=mock_model,
            dataloader=dummy_dataloader,
            device=torch.device("cpu"),
            method="percentile",
            percentile=0.99,
            spatial_mask_margin=0.15,
        )
        self.assertLessEqual(res["threshold"], 0.1)


if __name__ == "__main__":
    unittest.main()

