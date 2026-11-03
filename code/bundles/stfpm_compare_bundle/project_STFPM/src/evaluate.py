"""Evaluation script for testing STFPM model performance on 3D printer defect test sets."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import yaml
import torch
from torchmetrics.classification import BinaryAUROC

# Ensure project root is in sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictor import Printer3DDefectPredictor, PredictionResult


def load_yaml(path: Path) -> dict:
    """Loads a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_images_from_dir(directory: Path) -> List[Path]:
    """Recursively collects image file paths from a directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if not directory.exists():
        return []
    return sorted([p for p in directory.rglob("*") if p.suffix.lower() in extensions and p.is_file()])


def evaluate_samples(
    predictor: Printer3DDefectPredictor,
    image_paths: List[Path],
    expected_anomaly: bool,
    threshold: float,
    batch_size: int = 16,
    spatial_mask_margin: Optional[float] = None,
) -> Tuple[List[PredictionResult], List[float], List[int]]:
    """Evaluates a list of images against ground truth label."""
    results: List[PredictionResult] = []
    scores: List[float] = []
    targets: List[int] = []
    gt_label = 1 if expected_anomaly else 0

    total = len(image_paths)
    for start_idx in range(0, total, batch_size):
        chunk = image_paths[start_idx : start_idx + batch_size]
        batch_results = predictor.predict_batch(
            chunk,
            custom_threshold=threshold,
            spatial_mask_margin=spatial_mask_margin,
            fail_safe=True,
            batch_size=batch_size,
        )
        for res in batch_results:
            results.append(res)
            if res.decision != "ERROR":
                scores.append(res.anomaly_score)
                targets.append(gt_label)

    return results, scores, targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate STFPM model on 3D printer test images.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/stfpm_model.ckpt",
        help="Path to trained STFPM .ckpt file (default: outputs/checkpoints/stfpm_model.ckpt)",
    )
    parser.add_argument("--config-dataset", type=str, default="configs/printer3d.yaml", help="Dataset config path")
    parser.add_argument("--config-base", type=str, default="configs/base.yaml", help="Base config path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="결함 판정 임계값 (미지정 시 model_metadata.json의 비지도 임계값 자동 적용)",
    )
    parser.add_argument(
        "--spatial-mask-margin",
        type=float,
        default=None,
        help="외곽 베드 경계 마스킹 마진 (예: 0.05)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="연산 장치 (cuda 또는 cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="평가 배치 크기 (하드웨어 제약 최대 16)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"❌ 에러: 체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")
        sys.exit(1)

    cfg_data = load_yaml(Path(args.config_dataset))
    batch_size = min(max(1, args.batch_size), 16)

    # 1. 고속 STFPM 추론기 초기화
    predictor = Printer3DDefectPredictor(
        checkpoint_path=ckpt_path,
        device=args.device,
        spatial_mask_margin=args.spatial_mask_margin,
    )

    applied_threshold = args.threshold if args.threshold is not None else predictor.threshold

    print("=" * 65)
    print("📊 3D 프린터 STFPM 결함 탐지 모델 종합 성능 평가")
    print(f"  - 체크포인트       : {ckpt_path}")
    print(f"  - 적용 판정 임계값 : {applied_threshold:.4f}")
    print(f"  - 배치 크기        : {batch_size}")
    print(f"  - 공간 마스킹 마진 : {predictor.spatial_mask_margin}")
    print("=" * 65)

    data_root = Path(cfg_data.get("root", "data/printer3d"))
    normal_test_rel = cfg_data.get("normal_test_dir", "test/good")
    abnormal_rel = cfg_data.get("abnormal_dir", "test/defect")

    good_dir = data_root / normal_test_rel if normal_test_rel else None
    defect_dir = data_root / abnormal_rel if abnormal_rel else None

    good_images = collect_images_from_dir(good_dir) if good_dir else []
    defect_images = collect_images_from_dir(defect_dir) if defect_dir else []

    print(f"📁 테스트 데이터 수집:")
    print(f"  - 정상 테스트 이미지 (Good)  : {len(good_images)}장 ({good_dir})")
    print(f"  - 결함 테스트 이미지 (Defect): {len(defect_images)}장 ({defect_dir})\n")

    if not good_images and not defect_images:
        print("⚠️ 평가할 테스트 이미지가 없습니다.")
        return

    all_scores: List[float] = []
    all_targets: List[int] = []

    # 2. 정상 이미지 평가 (Label = 0)
    good_results, good_scores, good_targets = evaluate_samples(
        predictor=predictor,
        image_paths=good_images,
        expected_anomaly=False,
        threshold=applied_threshold,
        batch_size=batch_size,
        spatial_mask_margin=args.spatial_mask_margin,
    )
    all_scores.extend(good_scores)
    all_targets.extend(good_targets)

    # 3. 결함 이미지 평가 (Label = 1)
    defect_results, defect_scores, defect_targets = evaluate_samples(
        predictor=predictor,
        image_paths=defect_images,
        expected_anomaly=True,
        threshold=applied_threshold,
        batch_size=batch_size,
        spatial_mask_margin=args.spatial_mask_margin,
    )
    all_scores.extend(defect_scores)
    all_targets.extend(defect_targets)

    # 4. 혼동 행렬 및 분류 지표 산출
    tp = sum(1 for r in defect_results if r.is_anomaly and r.decision != "ERROR")
    fn = sum(1 for r in defect_results if not r.is_anomaly and r.decision != "ERROR")
    tn = sum(1 for r in good_results if not r.is_anomaly and r.decision != "ERROR")
    fp = sum(1 for r in good_results if r.is_anomaly and r.decision != "ERROR")
    defect_errors = sum(1 for r in defect_results if r.decision == "ERROR")
    good_errors = sum(1 for r in good_results if r.decision == "ERROR")
    total_errors = defect_errors + good_errors

    defect_total = tp + fn
    good_total = tn + fp
    total_valid = defect_total + good_total

    recall = (tp / defect_total) if defect_total > 0 else 0.0  # Defect Detection Rate
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    fpr = (fp / good_total) if good_total > 0 else 0.0
    tnr = (tn / good_total) if good_total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = ((tp + tn) / total_valid) if total_valid > 0 else 0.0

    auroc_str = "N/A"
    if len(set(all_targets)) > 1 and len(all_scores) > 1:
        try:
            auroc_metric = BinaryAUROC()
            auroc_val = float(auroc_metric(torch.tensor(all_scores), torch.tensor(all_targets)).item())
            auroc_str = f"{auroc_val:.4f}"
        except Exception as e:
            auroc_str = f"Error ({e})"

    good_score_mean = float(torch.tensor(good_scores).mean().item()) if good_scores else 0.0
    good_score_std = float(torch.tensor(good_scores).std().item()) if len(good_scores) > 1 else 0.0
    defect_score_mean = float(torch.tensor(defect_scores).mean().item()) if defect_scores else 0.0
    defect_score_std = float(torch.tensor(defect_scores).std().item()) if len(defect_scores) > 1 else 0.0

    print("=" * 65)
    print("📈 STFPM 성능 평가 결과 요약")
    print("=" * 65)
    print(f"  - AUROC                  : {auroc_str}")
    print(f"  - 결함 탐지율 (Recall)   : {recall * 100:.2f}% ({tp}/{defect_total})")
    print(f"  - 정밀도 (Precision)     : {precision * 100:.2f}% ({tp}/{tp + fp})")
    print(f"  - F1-Score               : {f1:.4f}")
    print(f"  - 오경보율 (FPR)         : {fpr * 100:.2f}% ({fp}/{good_total})")
    print(f"  - 정상 식별율 (TNR)      : {tnr * 100:.2f}% ({tn}/{good_total})")
    print(f"  - 전체 정확도 (Accuracy) : {accuracy * 100:.2f}% ({tp + tn}/{total_valid})")
    if total_errors > 0:
        print(f"  - 결측/오류 (Errors)     : {total_errors}장 (Good: {good_errors}, Defect: {defect_errors})")
    print("-" * 65)
    print("  [혼동 행렬 / Confusion Matrix]")
    print(f"                예측 정상(Good)   예측 결함(Defect)")
    print(f"  실제 정상      TN: {tn:<12} FP: {fp}")
    print(f"  실제 결함      FN: {fn:<12} TP: {tp}")
    if total_errors > 0:
        print(f"  결측/오류      Good: {good_errors:<10} Defect: {defect_errors}")
    print("-" * 65)
    print(f"  - 정상 이미지 평균 스코어: {good_score_mean:.4f} (± {good_score_std:.4f})")
    print(f"  - 결함 이미지 평균 스코어: {defect_score_mean:.4f} (± {defect_score_std:.4f})")
    print(f"  - 적용 임계값 (Threshold): {applied_threshold:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
