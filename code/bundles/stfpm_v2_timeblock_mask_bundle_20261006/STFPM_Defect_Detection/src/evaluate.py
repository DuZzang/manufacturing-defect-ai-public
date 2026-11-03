"""Evaluation script for testing STFPM model performance on 3D printer defect test sets."""

import argparse
import gc
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
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
    min_filament_ratio: Optional[float] = None,
    aggregation_method: Optional[str] = None,
    topk_ratio: Optional[float] = None,
    alpha: Optional[float] = None,
    eval_layers: Optional[Sequence[str]] = None,
    keep_results: bool = False,
) -> Tuple[List[PredictionResult], List[float], List[int], Dict[str, int]]:
    """Evaluates a list of images against ground truth label."""
    results: List[PredictionResult] = []
    scores: List[float] = []
    targets: List[int] = []
    counts: Dict[str, int] = {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "errors": 0}
    gt_label = 1 if expected_anomaly else 0

    total = len(image_paths)
    tag = "Defect" if expected_anomaly else "Good"
    for start_idx in range(0, total, batch_size):
        chunk = image_paths[start_idx : start_idx + batch_size]
        batch_results = predictor.predict_batch(
            chunk,
            custom_threshold=threshold,
            spatial_mask_margin=spatial_mask_margin,
            min_filament_ratio=min_filament_ratio,
            aggregation_method=aggregation_method,
            topk_ratio=topk_ratio,
            alpha=alpha,
            eval_layers=eval_layers,
            fail_safe=True,
            batch_size=batch_size,
        )
        for res in batch_results:
            if res.decision == "ERROR":
                counts["errors"] += 1
            else:
                scores.append(res.anomaly_score)
                targets.append(gt_label)
                if expected_anomaly:
                    if res.is_anomaly:
                        counts["tp"] += 1
                    else:
                        counts["fn"] += 1
                else:
                    if res.is_anomaly:
                        counts["fp"] += 1
                    else:
                        counts["tn"] += 1
            if keep_results:
                results.append(res)
        del batch_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.01)  # Yield GPU slice to OS/DWM to prevent Windows TDR watchdog reset

        processed = min(start_idx + batch_size, total)
        if (start_idx // batch_size) % 20 == 0 or processed == total:
            print(f"  [{tag}] 평가 진행: {processed}/{total} 장 완료 ({(processed/total)*100:.1f}%)", flush=True)

    return results, scores, targets, counts


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
        "--aggregation-method",
        type=str,
        choices=["topk_mean", "mean", "dual_metric", "amax"],
        default=None,
        help="이상치 점수 집계 방식 ('topk_mean', 'mean', 'dual_metric', 'amax')",
    )
    parser.add_argument(
        "--topk-ratio",
        type=float,
        default=None,
        help="topk_mean 방식 적용 시 상위 픽셀 비율 (예: 0.05)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="dual_metric 방식 적용 시 peak(amax) 가중치 (기본: 0.2)",
    )
    parser.add_argument(
        "--eval-layers",
        nargs="*",
        default=None,
        help="평가/추론 시 피처 맵을 결합할 레이어 목록 (예: layer2 layer3)",
    )
    parser.add_argument(
        "--spatial-mask-margin",
        type=float,
        default=None,
        help="외곽 베드 경계 마스킹 마진 (예: 0.05)",
    )
    parser.add_argument(
        "--min-filament-ratio",
        type=float,
        default=None,
        help="최소 필라멘트 영역 비율 (예: 0.25 지정 시 25%% 미만 베드 이미지 ERROR 처리)",
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
    # Hidden flags for memory-isolated worker subprocess execution
    parser.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-paths-file", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-expected-anomaly", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out-file", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Worker subprocess branch: processes isolated chunk and exits cleanly
    if args.worker_mode:
        if not args.worker_paths_file or not args.worker_out_file:
            print("❌ Worker mode requires --worker-paths-file and --worker-out-file")
            sys.exit(1)

        with open(args.worker_paths_file, "r", encoding="utf-8") as f:
            chunk_paths = [Path(line.strip()) for line in f if line.strip()]

        batch_size = min(max(1, args.batch_size), 16)
        predictor = Printer3DDefectPredictor(
            checkpoint_path=args.checkpoint,
            device=args.device,
            spatial_mask_margin=args.spatial_mask_margin,
            min_filament_ratio=args.min_filament_ratio,
            aggregation_method=args.aggregation_method,
            topk_ratio=args.topk_ratio,
            alpha=args.alpha,
            eval_layers=args.eval_layers,
        )
        applied_threshold = args.threshold if args.threshold is not None else predictor.threshold

        _, scores, targets, counts = evaluate_samples(
            predictor=predictor,
            image_paths=chunk_paths,
            expected_anomaly=bool(args.worker_expected_anomaly),
            threshold=applied_threshold,
            batch_size=batch_size,
            spatial_mask_margin=args.spatial_mask_margin,
            min_filament_ratio=args.min_filament_ratio,
            aggregation_method=args.aggregation_method,
            topk_ratio=args.topk_ratio,
            alpha=args.alpha,
            eval_layers=args.eval_layers,
            keep_results=False,
        )

        with open(args.worker_out_file, "w", encoding="utf-8") as f:
            json.dump({"scores": scores, "targets": targets, "counts": counts}, f)
        sys.exit(0)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"❌ 에러: 체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")
        sys.exit(1)

    cfg_data = load_yaml(Path(args.config_dataset))
    batch_size = min(max(1, args.batch_size), 16)

    # 1. 고속 STFPM 추론기 초기화 (메타데이터 및 임계값 확인용)
    predictor = Printer3DDefectPredictor(
        checkpoint_path=ckpt_path,
        device=args.device,
        spatial_mask_margin=args.spatial_mask_margin,
        min_filament_ratio=args.min_filament_ratio,
        aggregation_method=args.aggregation_method,
        topk_ratio=args.topk_ratio,
        alpha=args.alpha,
        eval_layers=args.eval_layers,
    )

    applied_threshold = args.threshold if args.threshold is not None else predictor.threshold

    print("=" * 65)
    print("📊 3D 프린터 STFPM 결함 탐지 모델 종합 성능 평가")
    print(f"  - 체크포인트       : {ckpt_path}")
    print(f"  - 적용 판정 임계값 : {applied_threshold:.6f}")
    print(f"  - 집계 방식        : {predictor.aggregation_method}")
    print(f"  - Top-K 비율       : {predictor.topk_ratio * 100:.1f}%")
    print(f"  - Dual Alpha       : {predictor.alpha}")
    print(f"  - 대상 레이어      : {predictor.eval_layers or 'All configured layers'}")
    print(f"  - 배치 크기        : {batch_size}")
    print(f"  - 공간 마스킹 마진 : {predictor.spatial_mask_margin}")
    if predictor.min_filament_ratio is not None:
        print(f"  - 최소 필라멘트    : {predictor.min_filament_ratio * 100:.1f}%")
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

    def run_chunk_subprocess(chunk_paths: List[Path], expected_anomaly: bool, chunk_tag: str):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as f_paths:
            for p in chunk_paths:
                f_paths.write(f"{p}\n")
            paths_file = Path(f_paths.name)

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f_out:
            out_file = Path(f_out.name)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-mode",
            "--worker-paths-file", str(paths_file),
            "--worker-expected-anomaly", "1" if expected_anomaly else "0",
            "--worker-out-file", str(out_file),
            "--checkpoint", str(ckpt_path),
            "--threshold", str(applied_threshold),
            "--batch-size", str(batch_size),
        ]
        eff_agg = args.aggregation_method if args.aggregation_method else predictor.aggregation_method
        eff_topk = args.topk_ratio if args.topk_ratio is not None else predictor.topk_ratio
        eff_alpha = args.alpha if args.alpha is not None else predictor.alpha
        eff_layers = args.eval_layers if args.eval_layers is not None else predictor.eval_layers
        eff_margin = args.spatial_mask_margin if args.spatial_mask_margin is not None else predictor.spatial_mask_margin
        eff_min_filament = args.min_filament_ratio if args.min_filament_ratio is not None else predictor.min_filament_ratio

        if eff_agg:
            cmd.extend(["--aggregation-method", eff_agg])
        if eff_topk is not None:
            cmd.extend(["--topk-ratio", str(eff_topk)])
        if eff_alpha is not None:
            cmd.extend(["--alpha", str(eff_alpha)])
        if eff_layers:
            cmd.extend(["--eval-layers", *eff_layers])
        if eff_margin is not None:
            cmd.extend(["--spatial-mask-margin", str(eff_margin)])
        if eff_min_filament is not None:
            cmd.extend(["--min-filament-ratio", str(eff_min_filament)])
        if args.device:
            cmd.extend(["--device", args.device])

        import os
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"

        try:
            subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
            with open(out_file, "r", encoding="utf-8") as f:
                res_data = json.load(f)
            return res_data["scores"], res_data["targets"], res_data["counts"]
        except subprocess.CalledProcessError as e:
            print(f"❌ Subprocess failed in chunk '{chunk_tag}':\nSTDERR:\n{e.stderr}\nSTDOUT:\n{e.stdout}")
            raise
        finally:
            if paths_file.exists():
                paths_file.unlink()
            if out_file.exists():
                out_file.unlink()

    all_scores: List[float] = []
    all_targets: List[int] = []

    # 2. 정상 이미지 평가 (Label = 0)
    good_scores, good_targets, good_counts = run_chunk_subprocess(good_images, False, "Good")
    all_scores.extend(good_scores)
    all_targets.extend(good_targets)
    time.sleep(1.5)  # GPU context cooldown to allow Windows DirectX virtual driver to release

    # 3. 결함 이미지 평가 (Label = 1, 500장 단위 프로세스 격리로 GPU 가상화 안정성 보장)
    defect_scores: List[float] = []
    defect_targets: List[int] = []
    defect_counts = {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "errors": 0}
    chunk_size_defect = 500

    for d_start in range(0, len(defect_images), chunk_size_defect):
        d_chunk = defect_images[d_start : d_start + chunk_size_defect]
        sub_scores, sub_targets, sub_counts = run_chunk_subprocess(
            d_chunk, True, f"Defect ({d_start+1}-{min(d_start+chunk_size_defect, len(defect_images))})"
        )
        defect_scores.extend(sub_scores)
        defect_targets.extend(sub_targets)
        for k in defect_counts:
            defect_counts[k] += sub_counts[k]
        time.sleep(1.5)  # Allow Windows/WSL2 DirectX GPU handle to fully unbind

    all_scores.extend(defect_scores)
    all_targets.extend(defect_targets)

    # 4. 혼동 행렬 및 분류 지표 산출
    tp = defect_counts["tp"]
    fn = defect_counts["fn"]
    tn = good_counts["tn"]
    fp = good_counts["fp"]
    defect_errors = defect_counts["errors"]
    good_errors = good_counts["errors"]
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
