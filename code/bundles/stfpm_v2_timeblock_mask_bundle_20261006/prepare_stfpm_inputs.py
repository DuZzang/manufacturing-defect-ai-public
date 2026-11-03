"""Make the two STFPM inputs from the SAME photo list, so masking is the only difference.

Photo list: the "2단계 선별 + 보류 사진 활용" list (manifest cohort primary 2,652 + rescue 743 = 3,395), the same list
the ConvNeXt 3-seed comparison used (manifest sha256 44584974...).
For every photo: decode the original once, cut the fixed crop (rows 0-1000, cols 250-1250), then save
  raw/<label>/<name>.png     the crop as is
  masked/<label>/<name>.png  the crop with everything outside the saved mask painted RGB(124,116,104)
Both are lossless PNG from the same decoded pixels, so JPEG re-compression can't explain a difference.
Writes split_b3395.csv = folds.csv rows for these photos + raw_crop / masked_crop columns (for stfpm_timeblock.py).

usage (Colab): python prepare_stfpm_inputs.py --bundle /content/bundle --orig /content/orig --out /content/stfpm_inputs
       local check: python prepare_stfpm_inputs.py --self-test
"""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

GRAY = np.array((124, 116, 104), np.uint8)
MANIFEST_SHA = "44584974b78d92fdf81e8177ff759f7bcc9fce202a493605f0cd156f3a2cbb57"


def make(original_rgb, keep):
    crop = original_rgb[0:1000, 250:1250]
    masked = crop.copy(); masked[~keep] = GRAY
    return crop, masked


def self_test():
    img = np.random.default_rng(0).integers(0, 255, (1200, 1600, 3), dtype=np.uint8)
    keep = np.zeros((1000, 1000), bool); keep[:, :400] = True
    raw, masked = make(img, keep)
    assert raw.shape == (1000, 1000, 3) and np.array_equal(raw, img[0:1000, 250:1250])
    assert np.array_equal(masked[:, :400], raw[:, :400]) and (masked[:, 400:] == GRAY).all()
    print("self-test OK: same crop / inside unchanged / outside gray")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path); ap.add_argument("--orig", type=Path); ap.add_argument("--out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="local check: only the first N photos")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return
    man_path = a.bundle / "manifest.csv"
    assert hashlib.sha256(man_path.read_bytes()).hexdigest() == MANIFEST_SHA, "manifest differs from the ConvNeXt comparison list"
    man = {r["source"]: r for r in csv.DictReader(open(man_path, encoding="utf-8-sig")) if r["cohort"] in ("primary", "rescue")}
    assert len(man) == 3395
    for i, (src, r) in enumerate(sorted(man.items())[:a.limit]):
        name = Path(src).stem + ".png"
        rawp, maskp = a.out / "raw" / r["label"] / name, a.out / "masked" / r["label"] / name
        if rawp.exists() and maskp.exists():
            continue
        rawp.parent.mkdir(parents=True, exist_ok=True); maskp.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(a.orig / src) as im:
            rgb = np.asarray(im.convert("RGB"))
        keep = np.asarray(Image.open(a.bundle / r["mask_path"])) > 127
        raw, masked = make(rgb, keep)
        Image.fromarray(raw).save(rawp); Image.fromarray(masked).save(maskp)
        if (i + 1) % 500 == 0:
            print(f"inputs {i + 1}/{len(man)}", flush=True)
    rows = []
    for r in csv.DictReader(open(a.bundle / "folds.csv", encoding="utf-8-sig")):
        src = r["source"].replace("\\", "/")
        if src in man:
            name = f"{man[src]['label']}/{Path(src).stem}.png"
            rows.append({**r, "source": src, "raw_crop": f"raw/{name}", "masked_crop": f"masked/{name}"})
    with open(a.out / "split_b3395.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    per_fold = {k: sum(r["fold"] == k for r in rows) for k in "01234"}
    print("split_b3395.csv rows per fold", per_fold, "(each fold lists all 3,395 photos with its role)")


if __name__ == "__main__":
    main()
