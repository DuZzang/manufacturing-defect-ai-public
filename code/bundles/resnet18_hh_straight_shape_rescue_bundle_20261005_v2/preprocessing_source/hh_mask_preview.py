"""Can the 2-stage usable selector's own maps serve as the bed mask? (preview only, no training)

Copies the two maps computed inside teammate image_metrics() (preprocess_integrated.py lines 50-75, unmodified
original) and returns them as masks instead of ratios:
  bed     : pixel map, value < 70 or brownish (hue 10-60, sat 50-220, value 40-180)
  surface : 32x32 tiles that are neutral-bright (>= 65%), have <= 20% bed pixels, and robust spread <= 0.25
Applied on the model crop (rows 0-1000, cols 250-1250) like hh_crop. Board: usable frames with the most bed.

usage: python work/gate_compare/hh_mask_preview.py --output outputs/<new folder> [--n 24]
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
HH = ROOT / "outputs/gate_compare_20261004/hh_stage2_values.csv"
LABELS = ROOT / "outputs/process_phase_labels_boundary_v5_20261004/phase_labels_boundary_updated.json"


def hh_maps(crop_rgb):
    """crop_rgb uint8 HxWx3 -> (bed pixel mask, surface tile mask), both bool at crop size."""
    im = Image.fromarray(crop_rgb)
    im.thumbnail((640, 640), Image.Resampling.LANCZOS)
    gray_image = im.convert("L")
    gray = np.asarray(gray_image, dtype=float) / 255
    rgb = np.asarray(im, dtype=float) / 255
    hsv = np.asarray(im.convert("HSV"), dtype=float)
    hue, sat, val = hsv[..., 0] * 360 / 255, hsv[..., 1], hsv[..., 2]
    bed = (val < 70) | ((hue >= 10) & (hue <= 60) & (sat >= 50) & (sat <= 220) & (val >= 40) & (val <= 180))
    neutral = (gray >= 100 / 255) & ((rgb.max(2) - rgb.min(2)) / np.maximum(rgb.max(2), 1e-6) <= .25)
    surface = np.zeros(gray.shape, bool)
    for y in range(0, gray.shape[0] - 31, 32):
        for x in range(0, gray.shape[1] - 31, 32):
            s = np.s_[y:y + 32, x:x + 32]
            if neutral[s].mean() < .65 or bed[s].mean() > .20:
                continue
            q25, median, q75 = np.quantile(gray[s], [.25, .5, .75])
            if (q75 - q25) / (median + 1e-6) > .25:
                continue
            surface[s] = True
    size = (crop_rgb.shape[1], crop_rgb.shape[0])
    up = lambda m: cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    return up(bed), up(surface)


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    if a.output.exists():
        ap.error("give a NEW --output folder")
    src = {f"{f['class_name']}/{f['frame']}": f["source_path"] for f in json.load(open(LABELS, encoding="utf-8"))["frames"]}
    usable = [r for r in csv.DictReader(open(HH, encoding="utf-8-sig")) if r["crop_decision"] == "usable"]
    usable.sort(key=lambda r: -float(r["crop_bed_ratio"]))
    top = usable[: a.n * 4]
    pick = [top[i] for i in np.linspace(0, len(top) - 1, a.n).astype(int)]   # spread over the bed-heavy quarter
    a.output.mkdir(parents=True)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 15)
    S = 300
    canvas = Image.new("RGB", (3 * (S + 6) * 2, 40 + (a.n // 2) * (S + 26)), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    d.text((8, 10), "usable 중 베드 많은 사진 / 원본 크롭 | 현환님 베드 화소(빨강) | 현환님 출력물 칸 밖 회색", font=font, fill="black")
    for i, r in enumerate(pick):
        bgr = cv2.imdecode(np.fromfile(src[f"{r['label']}/{r['frame']}"], np.uint8), cv2.IMREAD_COLOR)
        crop = cv2.cvtColor(bgr[0:1000, 250:1250], cv2.COLOR_BGR2RGB)
        bed, surface = hh_maps(crop)
        red = crop.copy(); red[bed] = (.4 * red[bed] + .6 * np.array([255, 0, 0])).astype(np.uint8)
        gray = crop.copy(); gray[~surface] = (124, 116, 104)
        x0, y0 = (i % 2) * 3 * (S + 6), 40 + (i // 2) * (S + 26)
        for j, im in enumerate((crop, red, gray)):
            canvas.paste(Image.fromarray(im).resize((S, S)), (x0 + j * (S + 6), y0))
        d.text((x0 + 3, y0 + S + 3), f"{r['label']} #{r['frame']}  베드 {float(r['crop_bed_ratio']):.2f}  "
                                     f"출력물 칸 {surface.mean():.2f}", font=font, fill="black")
    canvas.save(a.output / "hh_mask_usable_preview.jpg", quality=85)
    print("saved")


if __name__ == "__main__":
    main()
