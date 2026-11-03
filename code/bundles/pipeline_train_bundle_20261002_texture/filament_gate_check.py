"""Reproduce the teammate's filament gatekeeper on every original photo and see what it removes.

Teammate code (STFPM_Defect_Detection/src/data/preprocessing.py): crop the fixed ROI [ymin 0, xmin 250,
ymax 1000, xmax 1250], convert BGR->gray, Otsu threshold (cv2.THRESH_BINARY + THRESH_OTSU), and call the
frame "no print" when the bright (>threshold) fraction is below min_filament_ratio (0.25 in their report).
cv2 is not installed here, so gray conversion and Otsu are re-implemented with numpy (same formulas).
"""
import csv
import os
import sys

import numpy as np
from PIL import Image

ROI = (0, 250, 1000, 1250)  # ymin, xmin, ymax, xmax


def otsu_foreground_ratio(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    levels = np.arange(256)
    w0 = np.cumsum(hist)                       # pixels <= t
    w1 = total - w0
    m0 = np.cumsum(hist * levels)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 / total - m0) ** 2 / (w0 * w1 / total)
    between[(w0 == 0) | (w1 == 0)] = -1
    t = int(np.argmax(between))                # cv2 picks the first maximum
    return float((gray > t).mean()), t


def filament_ratio(rgb_crop):
    """Bright-pixel (Otsu) fraction of an RGB uint8 crop; same as the teammate's check_filament_presence."""
    c = rgb_crop.astype(np.float32)
    # cv2 BGR2GRAY on a BGR image == 0.299 R + 0.587 G + 0.114 B, rounded
    gray = np.clip(np.rint(c[..., 0] * 0.299 + c[..., 1] * 0.587 + c[..., 2] * 0.114), 0, 255).astype(np.uint8)
    return otsu_foreground_ratio(gray)[0]


def check():
    # tiny self-check: half dark / half bright -> ratio 0.5
    g = np.zeros((10, 10), np.uint8); g[:, 5:] = 200
    r, t = otsu_foreground_ratio(g)
    assert abs(r - 0.5) < 1e-9 and 0 <= t < 200


def main():
    ROOT, OUT = sys.argv[1], sys.argv[2]   # folder with normal/ and underfill_50/, output csv
    check()
    rows = []
    for label in ("normal", "underfill_50"):
        for f in sorted(os.listdir(os.path.join(ROOT, label))):
            if not f.lower().endswith(".jpg"):
                continue
            rgb = np.asarray(Image.open(os.path.join(ROOT, label, f)).convert("RGB"))
            ratio = filament_ratio(rgb[ROI[0]:ROI[2], ROI[1]:ROI[3]])
            rows.append({"source": f"{label}\\{f}", "label": label, "frame_index": int(f[:8]),
                         "filament_ratio": round(ratio, 4)})
    with open(OUT, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows ->", OUT)


if __name__ == "__main__":
    main()
