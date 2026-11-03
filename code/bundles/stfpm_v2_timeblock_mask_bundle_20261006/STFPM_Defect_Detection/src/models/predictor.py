"""Standalone high-speed inference predictor for 3D printer anomaly detection using STFPM."""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from anomalib.models import Stfpm
from src.data.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MIN_FILAMENT_RATIO,
    DEFAULT_ROI_CROP,
    Printer3DPreprocessor,
    apply_spatial_patch_mask,
    validate_input_image,
)


@dataclass
class PredictionResult:
    """Dataclass storing defect classification results."""

    image_path: Optional[str]
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    decision: str  # "GOOD", "DEFECT", or "ERROR"
    error_message: Optional[str] = None

    def __str__(self) -> str:
        if self.decision == "ERROR":
            return (
                f"[{self.decision}] ⚠️ 오류 / ERROR\n"
                f"  - 파일: {self.image_path or 'In-memory ndarray'}\n"
                f"  - 상세: {self.error_message or '알 수 없는 입력 결측/오류'}"
            )

        status_icon = "❌ DEFECT (결함 감지)" if self.is_anomaly else "✅ GOOD (정상 상태)"
        return (
            f"[{self.decision}] {status_icon}\n"
            f"  - 파일: {self.image_path or 'In-memory ndarray'}\n"
            f"  - 이상치 점수 (Anomaly Score): {self.anomaly_score:.4f}\n"
            f"  - 판정 임계값 (Threshold)    : {self.threshold:.4f}\n"
            f"  - 판정 기준: 점수가 임계값 이상이면 결함으로 판단합니다."
        )


class Printer3DDefectPredictor:
    """Predictor for evaluating 3D print layer defects using STFPM."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        spatial_mask_margin: Optional[float] = None,
        min_filament_ratio: Optional[float] = None,
        aggregation_method: Optional[str] = None,
        topk_ratio: Optional[float] = None,
        alpha: Optional[float] = None,
        eval_layers: Optional[Sequence[str]] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

        # Determine compute device (priority: CUDA if available)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Locate model metadata
        if metadata_path is None:
            candidate_meta = self.checkpoint_path.parent / "model_metadata.json"
            if candidate_meta.exists():
                metadata_path = candidate_meta

        # Fallback defaults if metadata is unavailable
        self.threshold: float = 0.5
        self.image_size: Sequence[int] = DEFAULT_IMAGE_SIZE
        self.roi_crop: Optional[Sequence[int]] = DEFAULT_ROI_CROP
        self.spatial_mask_margin: float = float(spatial_mask_margin) if spatial_mask_margin is not None else 0.05
        self.min_filament_ratio: Optional[float] = min_filament_ratio
        self.aggregation_method: str = aggregation_method if aggregation_method is not None else "mean"
        self.topk_ratio: float = float(topk_ratio) if topk_ratio is not None else 0.05
        self.alpha: float = float(alpha) if alpha is not None else 0.2
        self.eval_layers: Optional[List[str]] = list(eval_layers) if eval_layers is not None else None

        # Load persisted threshold and preprocessing parameters from metadata
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.threshold = float(meta.get("image_threshold", meta.get("threshold", 0.5)))
                if "image_size" in meta and meta["image_size"] is not None:
                    self.image_size = tuple(meta["image_size"])
                if "roi_crop" in meta:
                    self.roi_crop = tuple(meta["roi_crop"]) if meta["roi_crop"] is not None else None
                if "spatial_mask_margin" in meta and spatial_mask_margin is None:
                    self.spatial_mask_margin = float(meta["spatial_mask_margin"])
                if "min_filament_ratio" in meta and min_filament_ratio is None:
                    self.min_filament_ratio = float(meta["min_filament_ratio"]) if meta["min_filament_ratio"] is not None else None
                if "aggregation_method" in meta and aggregation_method is None:
                    self.aggregation_method = str(meta["aggregation_method"])
                if "topk_ratio" in meta and topk_ratio is None:
                    self.topk_ratio = float(meta["topk_ratio"])
                if "alpha" in meta and alpha is None:
                    self.alpha = float(meta["alpha"])
                if "eval_layers" in meta and eval_layers is None:
                    self.eval_layers = list(meta["eval_layers"]) if meta["eval_layers"] is not None else None

        # Instantiate unified preprocessing pipeline (1:1 Square Crop + Native RGB)
        self.preprocessor = Printer3DPreprocessor(
            image_size=self.image_size,
            roi_crop=self.roi_crop,
            device="cpu",
            min_filament_ratio=self.min_filament_ratio,
        )

        # Load STFPM model from checkpoint (with fallback to Patchcore for legacy compatibility)
        self.is_stfpm = True
        try:
            self.model = Stfpm.load_from_checkpoint(str(self.checkpoint_path), weights_only=False)
        except Exception:
            try:
                from anomalib.models import Patchcore
                self.model = Patchcore.load_from_checkpoint(str(self.checkpoint_path), weights_only=False)
                self.is_stfpm = False
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint {self.checkpoint_path}: {e}")

        self.model.to(self.device)
        self.model.eval()

        # Fallback to checkpoint's internal threshold if metadata was absent
        if not metadata_path or not Path(metadata_path).exists():
            if hasattr(self.model, "image_threshold"):
                val = self.model.image_threshold
                if hasattr(val, "value"):
                    val = val.value
                if isinstance(val, torch.Tensor) and not torch.isnan(val).item():
                    self.threshold = float(val.cpu().item())
                elif isinstance(val, (int, float)):
                    self.threshold = float(val)
            elif hasattr(self.model, "post_processor") and self.model.post_processor is not None:
                pp = self.model.post_processor
                if hasattr(pp, "_image_threshold"):
                    val = pp._image_threshold
                    if isinstance(val, torch.Tensor) and not torch.isnan(val).item():
                        self.threshold = float(val.cpu().item())
                elif hasattr(pp, "image_threshold"):
                    val = pp.image_threshold
                    if isinstance(val, torch.Tensor) and not torch.isnan(val).item():
                        self.threshold = float(val.cpu().item())
                    elif isinstance(val, (int, float)):
                        self.threshold = float(val)

        # Disable model's internal pre_processor and post_processor
        # Prevents duplicate resizing/normalization and keeps raw STFPM anomaly scores matching saved threshold
        self.model.pre_processor = torch.nn.Identity()
        self.model.post_processor = torch.nn.Identity()


    @torch.no_grad()
    def _forward_model(
        self,
        batch_tensor: torch.Tensor,
        margin: Optional[float] = None,
        aggregation_method: Optional[str] = None,
        topk_ratio: Optional[float] = None,
        alpha: Optional[float] = None,
        eval_layers: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        """Executes high-speed direct PyTorch forward inference on STFPM model.

        Applies spatial boundary masking and robust anomaly score aggregation
        (mean, topk_mean, dual_metric, or amax) to separate diffuse texture defects
        from bed scratches and boundary noise.

        Args:
            batch_tensor: Preprocessed image tensor of shape (B, 3, H, W).
            margin: Spatial margin ratio to suppress along bed borders (e.g. 0.05).
            aggregation_method: Pooling method ('mean', 'topk_mean', 'dual_metric', 'amax').
            topk_ratio: Ratio of top pixels to average when method is 'topk_mean' (e.g. 0.05).
            alpha: Weight for amax in 'dual_metric' (alpha * amax + (1 - alpha) * mean).
            eval_layers: Optional subset of layers to combine (e.g. ['layer2', 'layer3']).

        Returns:
            1D Tensor of shape (B,) containing anomaly scores.
        """
        if self.model.training:
            self.model.eval()

        eff_margin = margin if margin is not None else self.spatial_mask_margin
        eff_method = aggregation_method if aggregation_method is not None else self.aggregation_method
        eff_topk = topk_ratio if topk_ratio is not None else self.topk_ratio
        eff_alpha = alpha if alpha is not None else self.alpha
        eff_layers = eval_layers if eval_layers is not None else self.eval_layers

        if self.is_stfpm:
            device = self.device
            try:
                param = next(self.model.model.parameters())
                device = param.device
                dtype = param.dtype
            except (StopIteration, Exception):
                dtype = torch.float32

            input_tensor = batch_tensor.to(device=device, dtype=dtype)

            # Check if custom layer subset is requested and different from model's layers
            model_layers = list(
                getattr(self.model.model, "layers", getattr(self.model, "hparams", {}).get("layers", []))
            )
            if eff_layers is not None and set(eff_layers) != set(model_layers) and hasattr(self.model.model, "teacher_model"):
                t_feats = self.model.model.teacher_model(input_tensor)
                s_feats = self.model.model.student_model(input_tensor)
                out_size = input_tensor.shape[-2:]
                anomaly_map = torch.ones(
                    input_tensor.shape[0], 1, out_size[0], out_size[1],
                    device=input_tensor.device, dtype=dtype
                )
                for l in eff_layers:
                    l_map = self.model.model.anomaly_map_generator.compute_layer_map(
                        t_feats[l], s_feats[l], out_size
                    )
                    anomaly_map = anomaly_map * l_map
                    del l_map
                del t_feats, s_feats
            else:
                out = self.model.model(input_tensor)
                anomaly_map = out.anomaly_map

            # Spatial boundary masking (suppress outer bed border artifacts)
            if eff_margin > 0.0:
                masked_map = apply_spatial_patch_mask(anomaly_map, margin_ratio=eff_margin)
                H, W = masked_map.shape[-2], masked_map.shape[-1]
                pad_y = int(H * eff_margin)
                pad_x = int(W * eff_margin)
                if pad_y > 0 or pad_x > 0:
                    valid_region = masked_map[..., pad_y : H - pad_y, pad_x : W - pad_x]
                else:
                    valid_region = masked_map
            else:
                valid_region = anomaly_map

            B = valid_region.shape[0]
            flat = valid_region.reshape(B, -1)

            if eff_method == "mean":
                scores = torch.mean(flat, dim=-1)
            elif eff_method == "topk_mean":
                k = max(1, int(round(flat.shape[-1] * eff_topk)))
                topk_vals, _ = torch.topk(flat, k=k, dim=-1)
                scores = torch.mean(topk_vals, dim=-1)
            elif eff_method == "dual_metric":
                amax_val = torch.amax(flat, dim=-1)
                mean_val = torch.mean(flat, dim=-1)
                scores = eff_alpha * amax_val + (1.0 - eff_alpha) * mean_val
            elif eff_method == "amax":
                scores = torch.amax(flat, dim=-1)
            else:
                scores = torch.mean(flat, dim=-1)

            if scores.ndim == 0:
                scores = scores.unsqueeze(0)
            return scores

        # Fallback for legacy PatchCore checkpoints
        model = self.model.model
        input_tensor = batch_tensor.to(device=model.memory_bank.device, dtype=model.memory_bank.dtype)
        if model.tiler:
            input_tensor = model.tiler.tile(input_tensor)

        features = model.feature_extractor(input_tensor)
        features = {layer: model.feature_pooler(feature) for layer, feature in features.items()}
        embedding = model.generate_embedding(features)

        if model.tiler:
            embedding = model.tiler.untile(embedding)

        batch_size, _, height, width = embedding.shape
        embedding = model.reshape_embedding(embedding)

        if model.memory_bank.size(0) == 0:
            raise ValueError("Memory bank is empty. Cannot provide anomaly scores")

        patch_scores, locations = model.nearest_neighbors(embedding=embedding, n_neighbors=1)
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))

        if eff_margin > 0.0:
            patch_scores_grid = patch_scores.reshape((batch_size, 1, height, width))
            masked_patch_scores_grid = apply_spatial_patch_mask(patch_scores_grid, margin_ratio=eff_margin)
            patch_scores = masked_patch_scores_grid.reshape((batch_size, -1))

        scores = model.compute_anomaly_score(patch_scores, locations, embedding)
        if scores.ndim == 0:
            scores = scores.unsqueeze(0)
        return scores

    def predict_batch(
        self,
        image_inputs: Sequence[Union[str, Path, np.ndarray]],
        custom_threshold: Optional[float] = None,
        spatial_mask_margin: Optional[float] = None,
        min_filament_ratio: Optional[float] = None,
        aggregation_method: Optional[str] = None,
        topk_ratio: Optional[float] = None,
        alpha: Optional[float] = None,
        eval_layers: Optional[Sequence[str]] = None,
        fail_safe: bool = True,
        batch_size: int = 16,
    ) -> List[PredictionResult]:
        """Evaluates a batch of images for 3D printer layer defects using STFPM.

        Args:
            image_inputs: Sequence of file paths or OpenCV BGR/RGB ndarrays.
            custom_threshold: Optional override for anomaly decision threshold.
            spatial_mask_margin: Optional override for bed boundary suppression margin ratio.
            min_filament_ratio: Optional override for minimum filament ratio gatekeeper.
            aggregation_method: Optional override for score aggregation method ('mean', 'topk_mean', 'dual_metric', 'amax').
            topk_ratio: Optional override for top-k ratio in 'topk_mean'.
            alpha: Optional override for alpha in 'dual_metric'.
            eval_layers: Optional override for layer subset (e.g. ['layer2', 'layer3']).
            fail_safe: If True, records ERROR PredictionResult on invalid inputs rather
                than raising exceptions.
            batch_size: Sub-batch size for forward inference (capped at 16 to respect
                GTX 1660 SUPER 6GB hardware limits).

        Returns:
            List of PredictionResult matching the order of image_inputs.
        """
        inputs_list = list(image_inputs)
        if not inputs_list:
            return []

        margin = spatial_mask_margin if spatial_mask_margin is not None else self.spatial_mask_margin
        eff_min_filament = min_filament_ratio if min_filament_ratio is not None else self.min_filament_ratio
        threshold_val = custom_threshold if custom_threshold is not None else self.threshold

        results: List[Optional[PredictionResult]] = [None] * len(inputs_list)
        valid_indices: List[int] = []
        valid_tensors: List[torch.Tensor] = []

        for idx, img_input in enumerate(inputs_list):
            img_path_str: Optional[str] = str(img_input) if isinstance(img_input, (str, Path)) else None

            # 1. Gatekeeper input validation
            is_valid, err_msg, validated_arr = validate_input_image(
                img_input, min_filament_ratio=eff_min_filament, roi_crop=self.roi_crop
            )
            if not is_valid or validated_arr is None:
                if fail_safe:
                    results[idx] = PredictionResult(
                        image_path=img_path_str,
                        is_anomaly=False,
                        anomaly_score=0.0,
                        threshold=threshold_val,
                        decision="ERROR",
                        error_message=err_msg,
                    )
                    continue
                raise ValueError(f"Gatekeeper validation failed for input {idx}: {err_msg}")

            # 2. Dedicated preprocessing pipeline
            try:
                img_tensor = self.preprocessor.preprocess_array(validated_arr, is_bgr=True)
                valid_indices.append(idx)
                valid_tensors.append(img_tensor)
            except Exception as e:
                if fail_safe:
                    results[idx] = PredictionResult(
                        image_path=img_path_str,
                        is_anomaly=False,
                        anomaly_score=0.0,
                        threshold=threshold_val,
                        decision="ERROR",
                        error_message=f"Preprocessing error: {e}",
                    )
                    continue
                raise

        # 3. Model batch inference with hardware-constrained chunking (batch size <= 16)
        if valid_tensors:
            chunk_size = min(max(1, batch_size), 16)
            try:
                scores_list: List[float] = []
                for start_idx in range(0, len(valid_tensors), chunk_size):
                    sub_tensors = valid_tensors[start_idx : start_idx + chunk_size]
                    batch_tensor = torch.cat(sub_tensors, dim=0)
                    sub_scores = self._forward_model(
                        batch_tensor,
                        margin=margin,
                        aggregation_method=aggregation_method,
                        topk_ratio=topk_ratio,
                        alpha=alpha,
                        eval_layers=eval_layers,
                    )
                    sub_scores_list = sub_scores.cpu().tolist()
                    if isinstance(sub_scores_list, float):
                        sub_scores_list = [sub_scores_list]
                    scores_list.extend(sub_scores_list)

                    del batch_tensor, sub_scores
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                        time.sleep(0.005)

                del valid_tensors
                for v_idx, score_val in zip(valid_indices, scores_list):
                    inp = inputs_list[v_idx]
                    p_str: Optional[str] = str(inp) if isinstance(inp, (str, Path)) else None
                    is_anomaly = bool(score_val >= threshold_val)
                    decision = "DEFECT" if is_anomaly else "GOOD"
                    results[v_idx] = PredictionResult(
                        image_path=p_str,
                        is_anomaly=is_anomaly,
                        anomaly_score=float(score_val),
                        threshold=threshold_val,
                        decision=decision,
                    )
            except Exception as e:
                if fail_safe:
                    for v_idx in valid_indices:
                        if results[v_idx] is None:
                            inp = inputs_list[v_idx]
                            p_str = str(inp) if isinstance(inp, (str, Path)) else None
                            results[v_idx] = PredictionResult(
                                image_path=p_str,
                                is_anomaly=False,
                                anomaly_score=0.0,
                                threshold=threshold_val,
                                decision="ERROR",
                                error_message=f"Model forward error: {e}",
                            )
                else:
                    raise

        return [r for r in results if r is not None]


    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
        custom_threshold: Optional[float] = None,
        spatial_mask_margin: Optional[float] = None,
        aggregation_method: Optional[str] = None,
        topk_ratio: Optional[float] = None,
        alpha: Optional[float] = None,
        eval_layers: Optional[Sequence[str]] = None,
        fail_safe: bool = True,
    ) -> PredictionResult:
        """Evaluates whether the given single image contains defects.

        Args:
            image_input: Path to image file or OpenCV BGR/RGB ndarray.
            custom_threshold: Optional override for anomaly decision threshold.
            spatial_mask_margin: Optional override for bed boundary suppression margin ratio.
            aggregation_method: Optional override for score aggregation method ('mean', 'topk_mean', 'dual_metric', 'amax').
            topk_ratio: Optional override for top-k ratio in 'topk_mean'.
            alpha: Optional override for alpha in 'dual_metric'.
            eval_layers: Optional override for layer subset (e.g. ['layer2', 'layer3']).
            fail_safe: If True, returns an ERROR PredictionResult on invalid inputs rather
                than raising exceptions.

        Returns:
            PredictionResult with classification decision, anomaly score, and threshold.
        """
        results = self.predict_batch(
            [image_input],
            custom_threshold=custom_threshold,
            spatial_mask_margin=spatial_mask_margin,
            aggregation_method=aggregation_method,
            topk_ratio=topk_ratio,
            alpha=alpha,
            eval_layers=eval_layers,
            fail_safe=fail_safe,
        )
        return results[0]
