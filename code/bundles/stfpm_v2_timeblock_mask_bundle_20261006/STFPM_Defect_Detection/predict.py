"""Root entry point for evaluating 3D printer defects on real images.

Usage Examples:
    # 1. 단일 이미지 결함 판별 (기본 체크포인트: outputs/checkpoints/stfpm_model.ckpt)
    python predict.py --image sample_print.jpg

    # 2. 체크포인트 명시 지정
    python predict.py --checkpoint outputs/checkpoints/stfpm_model.ckpt --image sample_print.jpg

    # 3. 특정 디렉터리 내 이미지 일괄 판별
    python predict.py --input-dir data/printer3d/test/defect

    # 4. 임계값 수동 지정 판별 (기본값은 model_metadata.json의 최적 임계값 사용)
    python predict.py --image sample.jpg --threshold 0.45

    # 5. 베드 테두리 노이즈 억제 (외곽 5% 마스킹 적용)
    python predict.py --image sample.jpg --spatial-mask-margin 0.05
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predictor import PredictionResult, Printer3DDefectPredictor


def get_image_files(input_dir: Path) -> List[Path]:
    """Retrieves all image files from the target directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in extensions and p.is_file()])


def main():
    parser = argparse.ArgumentParser(
        description="3D Printer Layer Anomaly Predictor (STFPM) - 판별 전용 실행 스크립트",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/stfpm_model.ckpt",
        help="학습된 STFPM 모델 체크포인트 파일 경로 (.ckpt, 기본값: outputs/checkpoints/stfpm_model.ckpt)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="결함을 판별할 단일 이미지 파일 경로",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="일괄 판별할 이미지가 담긴 디렉터리 경로",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="결함 판정 임계값 (지정하지 않을 경우 학습 시 산출된 최적 임계값 자동 적용)",
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
        help="피처 맵을 결합할 레이어 목록 (예: layer2 layer3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="일괄 판별 시 미니배치 크기 (기본값: 16, 하드웨어 제약 최대 16)",
    )
    parser.add_argument(
        "--spatial-mask-margin",
        type=float,
        default=None,
        help="베드 가장자리 이상치 억제 마진 비율 (예: 0.05 지정 시 외곽 5%% 제외)",
    )
    parser.add_argument(
        "--min-filament-ratio",
        type=float,
        default=None,
        help="베드 과다/비출력 이미지 제외를 위한 최소 필라멘트 영역 비율 (예: 0.25 지정 시 25%% 미만 베드 이미지를 결측/ERROR 처리)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="연산 장치 (기본: cuda 자동 감지, 또는 cpu)",
    )
    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error("반드시 --image 또는 --input-dir 중 하나를 지정해야 합니다.")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"❌ 에러: 체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")
        sys.exit(1)

    # Hardware constraint enforcement (GTX 1660 SUPER, 6GB VRAM: batch size <= 16)
    batch_size = min(max(1, args.batch_size), 16)
    if args.batch_size > 16:
        print(f"⚠️ 경고: 하드웨어 제약 규칙에 따라 배치 크기를 16으로 제한합니다. (입력값: {args.batch_size})")

    print("=" * 65)
    print("🔍 3D 프린터 출력물 결함 판별기 (Defect Predictor - STFPM)")
    print(f"  - 모델 체크포인트: {ckpt_path.name}")
    print(f"  - 미니배치 크기  : {batch_size}")
    print("=" * 65)

    # 1. 고속 독립 추론기 초기화
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
    print(f"  - 적용 판정 임계값: {applied_threshold:.6f} (이 값 이상이면 결함 판정)")
    print(f"  - 적용 집계 방식  : {predictor.aggregation_method}")
    print(f"  - 공간 마스킹 마진: {predictor.spatial_mask_margin}")
    if predictor.min_filament_ratio is not None:
        print(f"  - 최소 필라멘트  : {predictor.min_filament_ratio * 100:.1f}%\n")
    else:
        print()

    # 2. 대상 이미지 수집
    target_images: List[Path] = []
    if args.image:
        target_images.append(Path(args.image))
    elif args.input_dir:
        dir_path = Path(args.input_dir)
        if not dir_path.exists():
            print(f"❌ 에러: 지정한 디렉터리가 존재하지 않습니다: {dir_path}")
            sys.exit(1)
        target_images = get_image_files(dir_path)
        print(f"📁 디렉터리 탐색 완료: 총 {len(target_images)}개 이미지 감지됨\n")

    if not target_images:
        print("⚠️ 판별할 이미지가 없습니다.")
        return

    # 3. 판별 루프 실행 (미니배치 처리 및 Fail-safe 지원)
    results: List[PredictionResult] = []
    good_count = 0
    defect_count = 0
    error_count = 0

    total_images = len(target_images)
    for start_idx in range(0, total_images, batch_size):
        chunk = target_images[start_idx : start_idx + batch_size]
        try:
            batch_results = predictor.predict_batch(
                chunk,
                custom_threshold=args.threshold,
                spatial_mask_margin=args.spatial_mask_margin,
                min_filament_ratio=args.min_filament_ratio,
                aggregation_method=args.aggregation_method,
                topk_ratio=args.topk_ratio,
                alpha=args.alpha,
                eval_layers=args.eval_layers,
                fail_safe=True,
                batch_size=batch_size,
            )
        except Exception as e:
            batch_results = [
                PredictionResult(
                    image_path=str(p),
                    is_anomaly=False,
                    anomaly_score=0.0,
                    threshold=applied_threshold,
                    decision="ERROR",
                    error_message=f"Batch execution exception: {e}",
                )
                for p in chunk
            ]

        for offset, res in enumerate(batch_results):
            current_idx = start_idx + offset + 1
            results.append(res)
            img_name = Path(res.image_path).name if res.image_path else "unknown"

            if res.decision == "ERROR":
                error_count += 1
                icon = "⚠️ [결측 / ERROR ]"
                print(f"[{current_idx}/{total_images}] {icon} | {res.error_message} | {img_name}")
            elif res.is_anomaly:
                defect_count += 1
                icon = "❌ [결함 / DEFECT]"
                print(f"[{current_idx}/{total_images}] {icon} | Score: {res.anomaly_score:.4f} (Thresh: {res.threshold:.4f}) | {img_name}")
            else:
                good_count += 1
                icon = "✅ [정상 / GOOD  ]"
                print(f"[{current_idx}/{total_images}] {icon} | Score: {res.anomaly_score:.4f} (Thresh: {res.threshold:.4f}) | {img_name}")

    # 4. 종합 요약 출력
    valid_count = good_count + defect_count
    print("\n" + "=" * 65)
    print("📊 최종 판별 통계 요약")
    print(f"  - 전체 검사 대상: {len(target_images)}장")
    print(f"  - 정상 (Good)   : {good_count}장 ({good_count / max(valid_count, 1) * 100:.1f}%)")
    print(f"  - 결함 (Defect) : {defect_count}장 ({defect_count / max(valid_count, 1) * 100:.1f}%)")
    if error_count > 0:
        print(f"  - 결측/오류     : {error_count}장")
    print("=" * 65)


if __name__ == "__main__":
    main()
