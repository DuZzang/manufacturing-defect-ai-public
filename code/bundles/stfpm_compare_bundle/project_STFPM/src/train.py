"""Training script for 3D printer STFPM anomaly detection."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import yaml
import torch

# Ensure project root is in sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomalib.engine import Engine
from src.data.datamodule import build_datamodule
from src.data.preprocessing import apply_spatial_patch_mask
from src.models.stfpm_builder import build_stfpm_model


def load_yaml(path: Path) -> dict:
    """Loads a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sync_threshold_to_model(model, chosen_thresh: float) -> None:
    """Synchronizes calculated threshold into Anomalib PostProcessor and model buffers if present."""
    if hasattr(model, "post_processor") and model.post_processor is not None:
        pp = model.post_processor
        if hasattr(pp, "_image_threshold") and isinstance(pp._image_threshold, torch.Tensor):
            pp._image_threshold.copy_(
                torch.tensor(chosen_thresh, dtype=pp._image_threshold.dtype, device=pp._image_threshold.device)
            )
        elif hasattr(pp, "image_threshold"):
            try:
                pp.image_threshold = torch.tensor(chosen_thresh, dtype=torch.float32)
            except AttributeError:
                pass
    if hasattr(model, "image_threshold"):
        if hasattr(model.image_threshold, "value"):
            model.image_threshold.value = torch.tensor(chosen_thresh, dtype=torch.float32)
        else:
            try:
                model.image_threshold = torch.tensor(chosen_thresh, dtype=torch.float32)
            except AttributeError:
                pass


def compute_unsupervised_threshold(
    model,
    dataloader=None,
    device=None,
    method: str = "percentile",
    percentile: float = 0.99,
    sigma_factor: float = 3.0,
    fallback_threshold: float = 0.5,
    limit_batches: Optional[int] = None,
    spatial_mask_margin: Optional[float] = None,
) -> dict:
    """Computes an unsupervised statistical anomaly threshold from normal train images.

    Training is strictly unsupervised (using only normal 3D printer layers).
    Threshold is computed directly from Student-Teacher feature discrepancy scores:
      - 99.0th Percentile (Q_0.99): Target False Positive Rate = 1%
      - Statistical Upper Bound (mu + 3*sigma): mean(scores) + 3 * std(scores)

    When dataloader is None, falls back to patch-level memory bank distance if present.

    Args:
        model: Trained STFPM or Patchcore model instance.
        dataloader: Training dataloader containing normal images.
        device: Torch execution device.
        method: Threshold strategy ('percentile' for Q_0.99, or 'mu_3sigma').
        percentile: Target quantile for percentile method (default: 0.99).
        sigma_factor: Factor multiplied by std for mu_3sigma (default: 3.0).
        fallback_threshold: Default threshold if scores cannot be evaluated.
        limit_batches: Optional maximum number of batches to evaluate (for dry-run mode).
        spatial_mask_margin: Optional spatial margin ratio to suppress along bed borders.

    Returns:
        dict containing threshold value, method name, mu, sigma, and alternative thresholds.
    """
    image_scores = []
    if dataloader is not None:
        model_inner = model.model if hasattr(model, "model") else model
        model_inner.eval()
        dev = device if device is not None else next(model_inner.parameters()).device
        model_inner.to(dev)

        margin_val = float(spatial_mask_margin) if spatial_mask_margin is not None else 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break
                if hasattr(batch, "image"):
                    imgs = batch.image
                elif isinstance(batch, dict) and "image" in batch:
                    imgs = batch["image"]
                elif isinstance(batch, (list, tuple)):
                    imgs = batch[0]
                else:
                    imgs = batch

                if not isinstance(imgs, torch.Tensor):
                    imgs = torch.as_tensor(imgs)
                if imgs.ndim == 3:
                    imgs = imgs.unsqueeze(0)
                if imgs.shape[0] == 0:
                    continue

                imgs = imgs.to(device=dev, dtype=torch.float32)
                out = model_inner(imgs)

                if margin_val > 0.0 and hasattr(out, "anomaly_map") and out.anomaly_map is not None:
                    masked_map = apply_spatial_patch_mask(out.anomaly_map, margin_ratio=margin_val)
                    pred_scores = torch.amax(masked_map, dim=(-2, -1)).flatten()
                else:
                    if hasattr(out, "pred_score") and out.pred_score is not None:
                        pred_score_raw = out.pred_score
                    elif isinstance(out, dict) and "pred_score" in out:
                        pred_score_raw = out["pred_score"]
                    else:
                        pred_score_raw = out
                    pred_scores = torch.as_tensor(pred_score_raw).flatten()

                image_scores.extend(pred_scores.cpu().float().tolist())

    if len(image_scores) >= 1:
        scores_t = torch.tensor(image_scores, dtype=torch.float32)
        valid_scores = scores_t[torch.isfinite(scores_t)]
        if len(valid_scores) >= 1:
            mu = float(valid_scores.mean().item())
            sigma = float(valid_scores.std().item()) if len(valid_scores) > 1 else 0.0
            thresh_mu_3sigma = float(mu + sigma_factor * sigma)
            thresh_percentile = float(torch.quantile(valid_scores, torch.tensor(percentile)).item())
            chosen_thresh = thresh_percentile if method == "percentile" else thresh_mu_3sigma

            _sync_threshold_to_model(model, chosen_thresh)

            return {
                "threshold": chosen_thresh,
                "method": f"stfpm_unsupervised_{method}",
                "mu": mu,
                "sigma": sigma,
                "threshold_mu_3sigma": thresh_mu_3sigma,
                "threshold_percentile_99_0": thresh_percentile,
                "sample_count": len(valid_scores),
            }

    # Fallback to patch-level memory bank distance if image dataloader is unavailable (e.g. legacy/mock tests)
    memory_bank = getattr(model.model, "memory_bank", None) if hasattr(model, "model") else None
    if memory_bank is not None and hasattr(model.model, "nearest_neighbors"):
        with torch.no_grad():
            patch_scores, _ = model.model.nearest_neighbors(embedding=memory_bank, n_neighbors=2)
            dists = patch_scores[:, 1].float()
            valid_mask = torch.isfinite(dists) & (dists > 0.0)
            valid_dists = dists[valid_mask]
            if len(valid_dists) == 0:
                valid_dists = dists[torch.isfinite(dists)]

            if len(valid_dists) > 0:
                mu = float(valid_dists.mean().item())
                sigma = float(valid_dists.std().item()) if len(valid_dists) > 1 else 0.0
                if not torch.isfinite(torch.tensor(sigma)):
                    sigma = 0.0
                thresh_mu_3sigma = float(mu + sigma_factor * sigma)
                thresh_percentile = float(
                    torch.quantile(valid_dists, torch.tensor(percentile, device=valid_dists.device)).item()
                )
                chosen_thresh = thresh_percentile if method == "percentile" else thresh_mu_3sigma
                _sync_threshold_to_model(model, chosen_thresh)

                return {
                    "threshold": chosen_thresh,
                    "method": f"unsupervised_{method}",
                    "mu": mu,
                    "sigma": sigma,
                    "threshold_mu_3sigma": thresh_mu_3sigma,
                    "threshold_percentile_99_0": thresh_percentile,
                    "sample_count": len(valid_dists),
                }

    # Model fallback threshold if post_processor has image_threshold
    if hasattr(model, "post_processor") and hasattr(model.post_processor, "image_threshold"):
        val = model.post_processor.image_threshold
        if isinstance(val, torch.Tensor) and not torch.isnan(val).item():
            fallback_threshold = float(val.cpu().item())
        elif isinstance(val, (int, float)) and not (isinstance(val, float) and torch.isnan(torch.tensor(val))):
            fallback_threshold = float(val)

    if hasattr(model, "post_processor") and getattr(model, "model", None) and getattr(model.model, "memory_bank", None) is None:
        return {
            "threshold": 0.5,
            "method": "fallback_model",
            "mu": 0.0,
            "sigma": 0.0,
            "threshold_mu_3sigma": 0.5,
            "threshold_percentile_99_0": 0.5,
            "sample_count": 0,
        }

    return {
        "threshold": fallback_threshold,
        "method": "fallback_default",
        "mu": 0.0,
        "sigma": 0.0,
        "threshold_mu_3sigma": fallback_threshold,
        "threshold_percentile_99_0": fallback_threshold,
        "sample_count": 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train STFPM model on 3D printer normal images.")
    parser.add_argument("--config-base", type=str, default="configs/base.yaml", help="Base config path")
    parser.add_argument("--config-dataset", type=str, default="configs/printer3d.yaml", help="Dataset config path")
    parser.add_argument("--config-model", type=str, default="configs/stfpm_resnet18.yaml", help="Model config path")
    parser.add_argument(
        "--threshold-method",
        type=str,
        choices=["percentile", "mu_3sigma"],
        default="percentile",
        help="비지도 학습 통계 임계값 산출 방식 ('percentile': Q_0.99 분위수, 'mu_3sigma': mu + 3*sigma)",
    )
    parser.add_argument(
        "--spatial-mask-margin",
        type=float,
        default=None,
        help="외곽 베드 경계 마스킹 마진 (미지정 시 데이터 설정 파일 값 또는 0.0 사용)",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="학습에 사용할 최대 배치 수 제한 (테스트/디버그용, 예: 1)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="학습 에포크 수 (미지정 시 모델 설정 파일 값 사용, 최대 20 제한)",
    )
    args = parser.parse_args()

    cfg_base = load_yaml(Path(args.config_base))
    cfg_data = load_yaml(Path(args.config_dataset))
    cfg_model = load_yaml(Path(args.config_model))

    # Hardware constraint enforcement (GTX 1660 SUPER, 6GB VRAM)
    backbone = cfg_model.get("backbone", "resnet18")
    if backbone not in ("resnet18", "resnet34"):
        print(f"⚠️ 경고: 하드웨어 제약 규칙에 따라 백본을 'resnet18'로 고정합니다. (기존 설정: {backbone})")
        backbone = "resnet18"

    train_batch_size = min(cfg_data.get("train_batch_size", 16), 16)
    eval_batch_size = min(cfg_data.get("eval_batch_size", 16), 16)

    # Epoch limit enforcement (Rule 2: STFPM training epochs <= 20)
    config_epochs = cfg_model.get("max_epochs", 15)
    max_epochs = args.max_epochs if args.max_epochs is not None else config_epochs
    if max_epochs > 20:
        print(f"⚠️ 경고: 과적합 방지 규칙에 따라 에포크를 20으로 제한합니다. (기존 설정: {max_epochs})")
        max_epochs = 20

    image_size = cfg_data.get("image_size", [512, 512])
    roi_crop = cfg_data.get("roi_crop", [0, 200, 1050, 1400])
    spatial_mask_margin = (
        args.spatial_mask_margin
        if args.spatial_mask_margin is not None
        else float(cfg_data.get("spatial_mask_margin", 0.0))
    )

    learning_rate = float(cfg_model.get("learning_rate", 0.4))
    momentum = float(cfg_model.get("momentum", 0.9))
    weight_decay = float(cfg_model.get("weight_decay", 0.001))

    print("=" * 65)
    print("🚀 3D 프린터 정상 데이터 기반 STFPM 학생-교사 증류 학습 시작")
    print(f"  - 백본 아키텍처   : {backbone}")
    print(f"  - 피처 피라미드   : {cfg_model.get('layers', ['layer1', 'layer2', 'layer3'])}")
    print(f"  - 학습률 (LR)     : {learning_rate}, 모멘텀: {momentum}, 가중치 감쇠: {weight_decay}")
    print(f"  - 데이터 경로     : {cfg_data.get('root', 'data/printer3d')}")
    print(f"  - 입력 해상도     : {image_size}")
    print(f"  - ROI Crop 영역   : {roi_crop}")
    print(f"  - 최대 학습 에포크: {max_epochs}")
    print(f"  - 배치 크기       : Train={train_batch_size}, Eval={eval_batch_size}")
    if args.limit_train_batches is not None:
        print(f"  - 배치 제한 모드  : {args.limit_train_batches}개 배치만 학습/검증에 사용")
    print("=" * 65)

    # 1. 데이터모듈 빌드 (통합 전처리 파이프라인 적용)
    datamodule = build_datamodule(
        root=cfg_data.get("root", "data/printer3d"),
        normal_dir=cfg_data.get("normal_dir", "train/good"),
        normal_test_dir=cfg_data.get("normal_test_dir", None),
        abnormal_dir=cfg_data.get("abnormal_dir", "test/defect"),
        mask_dir=cfg_data.get("mask_dir", None),
        normal_split_ratio=cfg_data.get("normal_split_ratio", None),
        image_size=image_size,
        roi_crop=roi_crop,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=cfg_data.get("num_workers", 4),
    )

    # 2. STFPM 모델 빌드
    layers = cfg_model.get("layers", ["layer1", "layer2", "layer3"])
    model = build_stfpm_model(
        backbone=backbone,
        layers=layers,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # 3. Anomalib Engine 초기화 (CUDA 부재 시 CPU 자동 안전 폴백)
    checkpoints_dir = Path(cfg_base.get("checkpoints_dir", "outputs/checkpoints"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    requested_accel = cfg_base.get("accelerator", "gpu")
    if requested_accel == "gpu" and not torch.cuda.is_available():
        print("⚠️ CUDA 사용 불가 감지: accelerator를 'cpu'로 자동 전환합니다.")
        accel = "cpu"
    else:
        accel = requested_accel

    engine_kwargs = {
        "default_root_dir": cfg_base.get("output_root", "outputs"),
        "accelerator": accel,
        "devices": cfg_base.get("devices", 1),
        "max_epochs": max_epochs,
    }
    if args.limit_train_batches is not None:
        engine_kwargs["limit_train_batches"] = args.limit_train_batches
        engine_kwargs["limit_val_batches"] = args.limit_train_batches

    engine = Engine(**engine_kwargs)

    # 4. 학습 (학생-교사 피처 피라미드 매칭 증류)
    engine.fit(datamodule=datamodule, model=model)

    # 5. 비지도 통계 임계값 산출 (Q_0.99 또는 mu+3*sigma)
    dev = torch.device("cuda" if torch.cuda.is_available() and accel != "cpu" else "cpu")
    print("\n🔍 정상 이미지 기반 STFPM 비지도 통계 임계값 산출 중 (Target FPR = 1%)...")
    threshold_info = compute_unsupervised_threshold(
        model=model,
        dataloader=datamodule.train_dataloader(),
        device=dev,
        method=args.threshold_method,
        percentile=0.99,
        sigma_factor=3.0,
        limit_batches=args.limit_train_batches,
        spatial_mask_margin=spatial_mask_margin,
    )
    image_threshold = threshold_info["threshold"]

    # 6. 메타데이터 저장 (하드코딩 방지 및 추론 일관성 유지)
    metadata = {
        "model_name": "stfpm",
        "backbone": backbone,
        "layers": layers,
        "image_size": image_size,
        "roi_crop": roi_crop,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "image_threshold": image_threshold,
        "threshold": image_threshold,
        "threshold_method": threshold_info["method"],
        "unsupervised_threshold_mu_3sigma": threshold_info["threshold_mu_3sigma"],
        "unsupervised_threshold_percentile": threshold_info.get("threshold_percentile_99_0", image_threshold),
        "normal_image_score_mean": threshold_info["mu"],
        "normal_image_score_std": threshold_info["sigma"],
        "sample_count": threshold_info.get("sample_count", 0),
        "spatial_mask_margin": spatial_mask_margin,
        "max_epochs": max_epochs,
    }

    meta_file = checkpoints_dir / "model_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 7. 체크포인트 저장
    ckpt_save_path = checkpoints_dir / "stfpm_model.ckpt"
    engine.trainer.save_checkpoint(str(ckpt_save_path))

    print("\n" + "=" * 65)
    print("✅ STFPM 학습 완료!")
    print(f"  - 모델 체크포인트 저장 경로: {ckpt_save_path}")
    print("  - 비지도 통계 임계값 산출 결과 (정상 이미지 레벨 스코어 기반):")
    print(f"    * 정상 이미지 평균 (mu): {threshold_info['mu']:.4f}, 표준편차 (sigma): {threshold_info['sigma']:.4f}")
    print(f"    * mu + 3*sigma: {threshold_info['threshold_mu_3sigma']:.4f}")
    if "threshold_percentile_99_0" in threshold_info:
        print(f"    * 99.0th 백분위수 (Q_0.99): {threshold_info['threshold_percentile_99_0']:.4f}")
    print(f"  - 적용된 최적 판정 임계값 (Threshold): {image_threshold:.4f} (방식: {threshold_info['method']})")
    print(f"  - 평가에 사용된 정상 이미지 수: {threshold_info.get('sample_count', 0)}장")
    print(f"  - 메타데이터 저장 경로: {meta_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
