#!/usr/bin/env python3
"""GAN 50장 파일럿용 표면 패치 준비. 원본과 기존 분할표는 읽기만 한다.

사용 예:
  python prepare_data.py --original-root /content/originals \
      --bundle /content/grid_bundle --out /content/gan_pilot_data
  python prepare_data.py --original-zip /content/originals.zip \
      --bundle /content/dinov2_grid_bundle.zip --out /content/gan_pilot_data

fold 0의 val은 시험, fold 4의 val은 내부 검증, 양쪽 buffer는 제외한다.
각 프레임의 1000x1000 고정 크롭에서 기존 베드 지도로 계산한 512x512
후보의 출력물 추정 비율이 최소 0.995인 곳을 고르고 원본 픽셀을 보존해
256x256으로 축소한다. 마스크는 좌표 선택에만 쓰며 이미지를 칠하지 않는다.

자동 마스크와 면적 기준은 임시 휴리스틱이다. 실제 언더필 틈, 과노출,
흔들림, 베드 유입을 사람이 확인해야 하며 정답 마스크가 아니다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TEST_FOLD = "0"
VALIDATION_FOLD = "4"
BUFFER_FRAMES = 30
MASK_MIN = 0.995
ROI_LEFT, ROI_TOP, ROI_SIZE = 250, 0, 1000
PATCH_SIZE, OUTPUT_SIZE, GRID_STEP = 512, 256, 32
NORMAL_LIMIT, DEFECT_LIMIT = 500, 50
LABELS = ("normal", "underfill_50")


class InputStore:
    """폴더와 ZIP을 같은 방식으로 읽는다. ZIP을 풀거나 수정하지 않는다."""

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve(strict=True)
        self.zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.zip is None and not self.path.is_dir():
            raise ValueError(f"입력은 폴더 또는 ZIP이어야 합니다: {self.path}")
        self.names = set(self.zip.namelist()) if self.zip else None

    def read(self, name: str) -> bytes:
        name = name.replace("\\", "/").lstrip("/")
        if ".." in Path(name).parts:
            raise ValueError(f"부적절한 경로: {name}")
        if self.zip:
            if name in self.names:
                return self.zip.read(name)
            matches = [n for n in self.names if n.endswith("/" + name)]
            if len(matches) != 1:
                raise FileNotFoundError(f"ZIP에서 {name}을 유일하게 찾지 못함: {self.path}")
            return self.zip.read(matches[0])
        path = (self.path / name).resolve()
        if not path.is_relative_to(self.path):
            raise ValueError(f"입력 루트 밖 경로: {name}")
        return path.read_bytes()

    def close(self) -> None:
        if self.zip:
            self.zip.close()


def read_csv(store: InputStore, name: str) -> list[dict[str, str]]:
    text = store.read(name).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def partition(fold_rows: list[dict]) -> dict[str, dict]:
    per_fold: dict[str, dict[str, dict]] = {TEST_FOLD: {}, VALIDATION_FOLD: {}}
    for row in fold_rows:
        fold = row["fold"]
        if fold not in per_fold:
            continue
        source = row["source"].replace("\\", "/")
        if source in per_fold[fold]:
            raise ValueError(f"중복된 fold/source: {fold}, {source}")
        per_fold[fold][source] = row
    if set(per_fold[TEST_FOLD]) != set(per_fold[VALIDATION_FOLD]):
        raise ValueError("fold 0과 fold 4의 원본 목록이 다릅니다.")
    result = {}
    for source, t in per_fold[TEST_FOLD].items():
        v = per_fold[VALIDATION_FOLD][source]
        if t["label"] != v["label"] or t["frame_index"] != v["frame_index"]:
            raise ValueError(f"fold 간 원본 정보가 다릅니다: {source}")
        tr, vr = t["role"], v["role"]
        if tr == "val" and vr == "val":
            raise ValueError(f"시험과 검증이 겹칩니다: {source}")
        if tr == "val":
            split = "test"
        elif vr == "val":
            split = "val"
        elif tr == "buffer" or vr == "buffer":
            split = "buffer"
        elif tr == vr == "train":
            split = "train"
        else:
            raise ValueError(f"알 수 없는 fold 역할: {source} {tr} {vr}")
        result[source] = {
            "source": source, "label": t["label"], "frame_index": int(t["frame_index"]),
            "fold0_role": tr, "fold4_role": vr, "split": split,
        }

    # 원본 프레임 순번 기준으로 학습/검증/시험 사이 완충 간격을 실제 확인한다.
    for label in LABELS:
        groups = {split: sorted(r["frame_index"] for r in result.values()
                                 if r["label"] == label and r["split"] == split)
                  for split in ("train", "val", "test")}
        if any(not group for group in groups.values()):
            raise ValueError(f"빈 분할이 있습니다: {label}")
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            # 2천 장 규모에서 명확하고 빠른 투 포인터 거리 계산.
            a, b = groups[left], groups[right]
            i = j = 0
            nearest = float("inf")
            while i < len(a) and j < len(b):
                nearest = min(nearest, abs(a[i] - b[j]))
                if a[i] < b[j]:
                    i += 1
                else:
                    j += 1
            if nearest <= BUFFER_FRAMES:
                raise ValueError(f"{label} {left}/{right} 최소 간격 {nearest} <= {BUFFER_FRAMES}")
    return result


def best_window(mask_bytes: bytes) -> tuple[int, int, float] | None:
    with Image.open(io.BytesIO(mask_bytes)) as image:
        mask = np.asarray(image.convert("L")) >= 128
    if mask.shape != (ROI_SIZE, ROI_SIZE):
        raise ValueError(f"베드 지도의 크기가 예상과 다릅니다: {mask.shape}")
    # 기본 중심 위치를 포함한 고정 간격 격자. 모든 픽셀 위치 완전탐색이 아니다.
    positions = sorted(set(range(0, ROI_SIZE - PATCH_SIZE + 1, GRID_STEP)) |
                       {0, (ROI_SIZE - PATCH_SIZE) // 2, ROI_SIZE - PATCH_SIZE})
    if mask.all():
        xy = (ROI_SIZE - PATCH_SIZE) // 2
        return xy, xy, 1.0
    if int(mask.sum()) < int(MASK_MIN * PATCH_SIZE * PATCH_SIZE):
        return None
    integral = np.zeros((ROI_SIZE + 1, ROI_SIZE + 1), dtype=np.int32)
    integral[1:, 1:] = np.cumsum(np.cumsum(mask, axis=0, dtype=np.int32), axis=1, dtype=np.int32)
    centers = (ROI_SIZE - PATCH_SIZE) / 2
    valid = []
    for y in positions:
        for x in positions:
            count = int(integral[y + PATCH_SIZE, x + PATCH_SIZE] - integral[y, x + PATCH_SIZE]
                        - integral[y + PATCH_SIZE, x] + integral[y, x])
            coverage = count / (PATCH_SIZE * PATCH_SIZE)
            if coverage >= MASK_MIN:
                valid.append(((x - centers) ** 2 + (y - centers) ** 2, -coverage, y, x))
    if not valid:
        return None
    _, negative_coverage, y, x = min(valid)
    return x, y, -negative_coverage


def spread_sample(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    """촬영 순서로 정렬 후 같은 개수의 후보 구간에서 1개씩 뽑는다.

    후보 순위 구간이므로 실제 시간 간격이 균일하다는 보장은 없다.
    """
    rows = sorted(rows, key=lambda r: (r["frame_index"], r["source"]))
    if len(rows) <= count:
        return rows
    selected = []
    for index in range(count):
        start = index * len(rows) // count
        end = (index + 1) * len(rows) // count
        selected.append(rows[rng.randrange(start, end)])
    return selected


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(args: argparse.Namespace) -> None:
    out = Path(args.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"기존 결과 보호: 출력 폴더가 비어 있지 않습니다: {out}")
    if args.smoke_limit is not None and args.smoke_limit < 1:
        raise ValueError("--smoke-limit은 1 이상이어야 합니다.")
    original = InputStore(Path(args.original_root or args.original_zip))
    bundle = InputStore(Path(args.bundle))
    try:
        folds_bytes, selections_bytes = bundle.read("folds.csv"), bundle.read("selections.csv")
        split_by_source = partition(read_csv(bundle, "folds.csv"))
        selections = read_csv(bundle, "selections.csv")
        selection_sources = [row["source"].replace("\\", "/") for row in selections]
        if len(selection_sources) != len(set(selection_sources)):
            raise ValueError("selections.csv에 같은 원본이 중복됩니다.")
        if set(selection_sources) != set(split_by_source):
            raise ValueError("selections.csv와 folds.csv의 원본 목록이 다릅니다.")
        all_rows = []
        for row in selections:
            source = row["source"].replace("\\", "/")
            meta = split_by_source[source].copy()
            if meta["label"] != row["label"] or meta["frame_index"] != int(row["frame_index"]):
                raise ValueError(f"선별표와 분할표의 라벨·프레임 불일치: {source}")
            meta.update(stage2=row["stage2"], stage2_reason=row.get("stage2_reason", ""),
                        bedmap_mask=row["bedmap_mask"], bedmap_area=row.get("bedmap_area", ""))
            all_rows.append(meta)

        counts = Counter((r["split"], r["label"], "all") for r in all_rows)
        grouped = {(split, label): [] for split in ("train", "val", "test") for label in LABELS}
        scan_complete = True
        for row in all_rows:
            if row["split"] == "buffer":
                continue
            if row["stage2"] != "1":
                continue
            key = (row["split"], row["label"])
            counts[(*key, "stage2")] += 1
            try:
                window = best_window(bundle.read(row["bedmap_mask"]))
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"마스크 판독 오류: {row['source']}") from exc
            if window is None:
                continue
            x, y, coverage = window
            row.update(x_roi=x, y_roi=y, mask_ratio=coverage)
            grouped[key].append(row)
            counts[(*key, "patch_eligible")] += 1
            if args.smoke_limit is not None and all(len(group) >= args.smoke_limit for group in grouped.values()):
                scan_complete = False
                break

        limit = args.smoke_limit
        target_defect = limit if limit is not None else DEFECT_LIMIT
        target_normal = limit if limit is not None else NORMAL_LIMIT
        if len(grouped[("train", "underfill_50")]) < target_defect:
            raise ValueError(f"훈련 언더필 후보가 {target_defect}장보다 적습니다: "
                             f"{len(grouped[('train', 'underfill_50')])}")
        rng = random.Random(args.seed)
        selected = {}
        for key, rows in grouped.items():
            if key == ("train", "underfill_50"):
                selected[key] = spread_sample(rows, target_defect, rng)
            elif key == ("train", "normal"):
                selected[key] = spread_sample(rows, target_normal, rng)
            elif limit is not None:
                selected[key] = spread_sample(rows, limit, rng)
            else:
                selected[key] = sorted(rows, key=lambda r: (r["frame_index"], r["source"]))
            if not selected[key]:
                raise ValueError(f"선택된 사진이 없습니다: {key}")

        membership_fields = ["source", "label", "frame_index", "fold0_role", "fold4_role", "split"]
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "membership_all4000.csv", sorted(split_by_source.values(),
                  key=lambda r: (r["label"], r["frame_index"])), membership_fields)

        manifest_fields = ["source", "label", "frame_index", "split", "stage2", "stage2_reason",
                           "bedmap_mask", "bedmap_area", "x_roi", "y_roi", "x_original", "y_original",
                           "patch_original_px", "output_px", "mask_ratio", "original_sha256", "output_path"]
        emitted = []
        for (split, label), rows in selected.items():
            for row in rows:
                data = original.read(row["source"])
                with Image.open(io.BytesIO(data)) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    if image.size != (1600, 1200):
                        raise ValueError(f"예상과 다른 원본 크기 {image.size}: {row['source']}")
                    x, y = ROI_LEFT + row["x_roi"], ROI_TOP + row["y_roi"]
                    patch = image.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                    patch = patch.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS)
                    output_split = "train_real" if split == "train" else split
                    output_label = "underfill" if label == "underfill_50" else label
                    relative = Path("data") / output_split / output_label / (Path(row["source"]).stem + ".png")
                    destination = out / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    patch.save(destination, format="PNG")
                record = {key: row[key] for key in ("source", "label", "frame_index", "split", "stage2",
                          "stage2_reason", "bedmap_mask", "bedmap_area", "x_roi", "y_roi", "mask_ratio")}
                record.update(x_original=x, y_original=y, patch_original_px=PATCH_SIZE, output_px=OUTPUT_SIZE,
                              mask_ratio=f"{row['mask_ratio']:.6f}", original_sha256=sha256(data),
                              output_path=relative.as_posix())
                emitted.append(record)
            print(f"{split}/{label}: {len(rows)}개 저장", flush=True)

        sources = [r["source"] for r in emitted]
        if len(sources) != len(set(sources)):
            raise RuntimeError("동일한 원본이 두 번 저장됐습니다.")
        for split in ("train", "val", "test"):
            write_csv(out / f"manifest_{split}.csv", [r for r in emitted if r["split"] == split], manifest_fields)
        write_csv(out / "manifest_all.csv", emitted, manifest_fields)
        audit_rows = []
        for split in ("train", "val", "test", "buffer"):
            for label in LABELS:
                total = counts[(split, label, "all")]
                stage2 = counts[(split, label, "stage2")]
                eligible = counts[(split, label, "patch_eligible")]
                output_count = len(selected.get((split, label), []))
                audit_rows.append(dict(split=split, label=label, scan_complete=scan_complete,
                    total_frames=total, stage2_pass=stage2,
                    patch_eligible=eligible, exported=output_count,
                    stage2_pass_rate=round(stage2 / total, 6) if scan_complete and total and split != "buffer" else "",
                    patch_eligible_rate=round(eligible / total, 6) if scan_complete and total and split != "buffer" else ""))
        write_csv(out / "audit_counts.csv", audit_rows,
                  ["split", "label", "scan_complete", "total_frames", "stage2_pass", "patch_eligible", "exported",
                   "stage2_pass_rate", "patch_eligible_rate"])
        audit = {
            "purpose": "exploratory GAN pilot; one print per class, no independent external test",
            "smoke_test": limit is not None, "smoke_limit": limit, "scan_complete": scan_complete, "seed": args.seed,
            "source": str(original.path), "bundle": str(bundle.path),
            "folds_csv_sha256": sha256(folds_bytes), "selections_csv_sha256": sha256(selections_bytes),
            "test_fold": int(TEST_FOLD), "validation_fold": int(VALIDATION_FOLD),
            "buffer": "union of fold0 and fold4 buffer; per-label minimum inter-split distance verified >30",
            "stage2": "v3.2 stored selection; original stage2 assessed whole top 85%, not the patch",
            "roi_xyxy_original": [250, 0, 1250, 1000], "patch_original_px": PATCH_SIZE,
            "patch_search": "grid stride 32, include central and endpoint; select eligible closest to ROI center",
            "mask_threshold": MASK_MIN, "mask_use": "coordinate selection only; pixels are not masked",
            "output_px": OUTPUT_SIZE, "interpolation": "Pillow LANCZOS", "color": "RGB",
            "train_defect_count": len(selected[("train", "underfill_50")]),
            "train_normal_count": len(selected[("train", "normal")]),
            "sampling": "chronological candidate rank bins; one seeded random sample per bin",
            "warnings": ["automatic mask is not ground truth; manual visual review required",
                         "only one print per class; same-print time split is not independent output",
                         "underfill_50 is a printing-condition label, not per-frame visible-defect ground truth",
                         "patch and 256 resize differ from prior 1000-to-512 classifier input",
                         "stage2 and 0.995 patch selection may exclude partial-surface frames",
                         "GAN train uses only train_real underfill; val is for tuning; test is never used for tuning",
                         "in smoke mode candidate scan may stop early; audit pass rates are incomplete and not research results"]}
        with (out / "audit_settings.json").open("w", encoding="utf-8") as stream:
            json.dump(audit, stream, ensure_ascii=False, indent=2)
        print(f"완료: {out}", flush=True)
    finally:
        original.close()
        bundle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--original-root", help="normal/ 및 underfill_50/를 포함한 원본 폴더")
    source.add_argument("--original-zip", help="같은 폴더 구조의 원본 ZIP")
    parser.add_argument("--bundle", required=True, help="selections.csv, folds.csv, masks를 포함한 폴더 또는 ZIP")
    parser.add_argument("--out", required=True, help="새 결과 폴더. 기존 비어 있지 않은 폴더에는 쓰지 않음")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-limit", type=int, help="코드 검증용 분할·클래스별 상한. 연구 결과로 사용 금지")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        print(f"데이터 준비 실패: {exc}", file=sys.stderr)
        raise
