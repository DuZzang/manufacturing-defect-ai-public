"""STFPM 분할 비교 입력 준비: 원욱님 방식(정상 5장 간격) 분할 1개 + 원본 크롭 PNG.

분할 (팀원 프로젝트 README: "균등 간격(Stride 5)", 학습 정상 1,600 / 시험 정상 400)
  정상 2,000장을 프레임 순으로 정렬해 순번 % 5 == OFFSET 인 400장 = 시험 정상, 나머지 1,600장 = 학습 후보.
  OFFSET은 README에 없다 -> 기본 0으로 두고 기록한다 (어느 1장을 빼든 이웃 1~4장이 학습에 남는 구조는 같다).
  학습 = 학습 후보 중 2단계 v3.2 통과 (시간 블록 실행과 같은 선별). 미통과는 role 'train_unselected'로 제외.
  시험 = 시험 정상 400 + 언더필 2,000 전부 (비지도라 언더필은 학습에 안 씀).
입력은 시간 블록 실행(s2_raw_mean)과 같은 고정 크롭 PNG. PNG는 무손실이라 압축 수준을 낮춰도 픽셀은 같다 (저장만 빨라짐).
split_stride5.csv 는 stfpm_timeblock.py 형식 (fold 0 하나).

usage (Colab): python prepare_stfpm_stride5.py --grid /content/grid --orig /content/orig --out /content/stfpm_inputs
       local check: python prepare_stfpm_stride5.py --self-test --grid <grid bundle>
"""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

OFFSET = 0


def split_rows(sel):
    normals = sorted((r for r in sel if r["label"] == "normal"), key=lambda r: int(r["frame_index"]))
    test = {r["source"] for i, r in enumerate(normals) if i % 5 == OFFSET}
    rows = []
    for r in sel:
        if r["label"] != "normal" or r["source"] in test:
            role = "val"
        else:
            role = "train" if int(r["stage2"]) else "train_unselected"
        rows.append(dict(fold=0, role=role, source=r["source"], label=r["label"], label_id=r["label_id"],
                         frame_index=r["frame_index"], stride_test_normal=int(r["source"] in test),
                         raw_crop=f"raw/{r['label']}/{Path(r['source']).stem}.png"))
    return rows


def check(rows):
    n = [r for r in rows if r["label"] == "normal"]
    test = sorted(int(r["frame_index"]) for r in n if r["role"] == "val")
    train = sorted(int(r["frame_index"]) for r in n if r["role"] != "val")
    assert len(test) == 400 and len(train) == 1600
    assert all(r["role"] == "val" for r in rows if r["label"] != "normal")
    # 시험 정상마다 학습 후보 이웃이 1~4프레임 안에 있다 = 이 분할의 특징 (시간 블록은 31프레임 이상 떨어짐)
    gap = [min(abs(t - x) for x in train) for t in test]
    return dict(test_normal=400, train_candidates=1600, train_stage2=sum(r["role"] == "train" for r in rows),
                val_total=sum(r["role"] == "val" for r in rows), max_gap_to_train=max(gap), mean_gap=sum(gap) / len(gap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=Path, required=True); ap.add_argument("--orig", type=Path); ap.add_argument("--out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    sel = list(csv.DictReader(open(a.grid / "selections.csv", encoding="utf-8-sig")))
    assert len(sel) == 4000
    rows = split_rows(sel)
    info = check(rows)
    print("분할:", info, flush=True)
    if a.self_test:
        assert info["max_gap_to_train"] <= 4
        print("self-test OK"); return
    for i, r in enumerate(rows):
        p = a.out / r["raw_crop"]
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(a.orig / r["source"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        Image.fromarray(rgb[0:1000, 250:1250]).save(p, compress_level=1)
        if (i + 1) % 500 == 0:
            print(f"inputs {i + 1}/4000", flush=True)
    with open(a.out / "split_stride5.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    print("split_stride5.csv 저장", flush=True)


if __name__ == "__main__":
    main()
