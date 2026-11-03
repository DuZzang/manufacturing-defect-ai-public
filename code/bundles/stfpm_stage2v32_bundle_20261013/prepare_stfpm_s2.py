"""STFPM 입력 준비: 2단계(v3.2) 정상 사진으로만 학습, 평가는 4,000장 전부.

입력 두 가지 (같은 원본을 한 번 읽어 무손실 PNG로 저장 -> JPEG 재압축 차이가 끼어들지 않음)
  raw/<label>/<name>.png      고정 크롭(행 0-1000, 열 250-1250) 그대로
  texture/<label>/<name>.png  같은 크롭, 질감 마스크 밖을 회색 RGB(124,116,104)  (ConvNeXt 질감 조건과 같은 마스크 파일)
split_s2v32.csv = folds.csv 4,000장 × 5fold + raw_crop / texture_crop 열.
  stfpm_timeblock.py는 role == 'train' 인 정상 사진만 학습하므로, 2단계 미통과 학습 사진은 role을 'train_unselected'로 바꿔 학습에서 뺀다.
  평가(role == 'val')는 선별과 무관하게 전부 남긴다 -> 모든 사진에 점수가 나온다.

usage (Colab): python prepare_stfpm_s2.py --grid /content/grid --texture /content/texpatch/texture --orig /content/orig --out /content/stfpm_inputs
       local check: python prepare_stfpm_s2.py --self-test
"""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

GRAY = np.array((124, 116, 104), np.uint8)
FOLDS_SHA = "364ed2f2bd22f919cb98e47a252fa85a2dcabba7b1bf913e0ad68b8b489f5f6b"


def make(original_rgb, keep):
    crop = original_rgb[0:1000, 250:1250]
    masked = crop.copy(); masked[~keep] = GRAY
    return crop, masked


def relabel(fold_rows, stage2):
    out = []
    for r in fold_rows:
        src = r["source"].replace("\\", "/")
        role = r["role"]
        if role == "train" and not stage2[src]:
            role = "train_unselected"
        out.append({**r, "source": src, "role": role})
    return out


def self_test():
    img = np.random.default_rng(0).integers(0, 255, (1200, 1600, 3), dtype=np.uint8)
    keep = np.zeros((1000, 1000), bool); keep[:, :400] = True
    raw, masked = make(img, keep)
    assert raw.shape == (1000, 1000, 3) and np.array_equal(raw, img[0:1000, 250:1250])
    assert np.array_equal(masked[:, :400], raw[:, :400]) and (masked[:, 400:] == GRAY).all()
    rows = relabel([dict(source="normal\\a.jpg", role="train"), dict(source="normal\\b.jpg", role="train"),
                    dict(source="normal\\c.jpg", role="val")], {"normal/a.jpg": 1, "normal/b.jpg": 0, "normal/c.jpg": 0})
    assert [r["role"] for r in rows] == ["train", "train_unselected", "val"]
    print("self-test OK: 같은 크롭 / 마스크 안 그대로 / 밖 회색 / 미통과 학습 사진만 제외")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=Path); ap.add_argument("--texture", type=Path)
    ap.add_argument("--orig", type=Path); ap.add_argument("--out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return
    assert hashlib.sha256((a.grid / "folds.csv").read_bytes()).hexdigest() == FOLDS_SHA, "folds.csv가 공통 분할과 다름"
    sel = {r["source"]: r for r in csv.DictReader(open(a.grid / "selections.csv", encoding="utf-8-sig"))}
    tex = {r["source"]: r for r in csv.DictReader(open(a.texture / "texture_manifest.csv", encoding="utf-8-sig"))}
    assert len(sel) == len(tex) == 4000 and sum(int(r["stage2"]) for r in sel.values()) == 3062
    for i, src in enumerate(sorted(sel)):
        t = tex[src]
        name = Path(src).stem + ".png"
        rawp, texp = a.out / "raw" / t["label"] / name, a.out / "texture" / t["label"] / name
        if rawp.exists() and texp.exists():
            continue
        rawp.parent.mkdir(parents=True, exist_ok=True); texp.parent.mkdir(parents=True, exist_ok=True)
        assert hashlib.sha256((a.orig / src).read_bytes()).hexdigest() == t["source_sha256"], src
        mask_path = a.texture / t["mask_path"]
        assert hashlib.sha256(mask_path.read_bytes()).hexdigest() == t["mask_sha256"], src
        with Image.open(a.orig / src) as im:
            rgb = np.asarray(im.convert("RGB"))
        keep = np.asarray(Image.open(mask_path)) > 127
        raw, masked = make(rgb, keep)
        Image.fromarray(raw).save(rawp); Image.fromarray(masked).save(texp)
        if (i + 1) % 500 == 0:
            print(f"inputs {i + 1}/4000", flush=True)
    stage2 = {s: int(r["stage2"]) for s, r in sel.items()}
    rows = relabel(list(csv.DictReader(open(a.grid / "folds.csv", encoding="utf-8-sig"))), stage2)
    for r in rows:
        name = f"{sel[r['source']]['label']}/{Path(r['source']).stem}.png"
        r.update(raw_crop=f"raw/{name}", texture_crop=f"texture/{name}")
    with open(a.out / "split_s2v32.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    for k in "01234":
        fold = [r for r in rows if r["fold"] == k]
        print(f"fold {k}: 학습 정상 {sum(r['role'] == 'train' and r['label'] == 'normal' for r in fold)}"
              f" (미통과로 뺀 정상 {sum(r['role'] == 'train_unselected' and r['label'] == 'normal' for r in fold)}),"
              f" 평가 {sum(r['role'] == 'val' for r in fold)}", flush=True)


if __name__ == "__main__":
    main()
