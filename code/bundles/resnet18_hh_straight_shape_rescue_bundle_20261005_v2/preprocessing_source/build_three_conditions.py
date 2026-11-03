"""Build the inputs of the three training conditions (no training here).

  1 selector        : 2-stage usable selector (hh_crop) usable frames, raw model crop          -> keep_cond1.csv
  2 selector + mask : same frames, bed painted gray with the selector's OWN bed pixel map     -> + masks_cond2.csv
  3 + shape v2      : 2, plus held frames recycled with the shape tracker v2 mask             -> keep_cond3.csv, masks_cond3.csv
                      (status SUPPORTED and print area on the model crop >= MIN_AREA)

Why the selector's bed map for condition 2: selection and masking then use one criterion (no third tool).
Why shape v2 only for held frames: it follows bed edges well but was dropped for full-surface frames (user review).
Masks are PNG on the model crop (rows 0-1000, cols 250-1250), 255 = keep. Used by train_pilot_classifier.py --mask-csv.

usage: python work/gate_compare/build_three_conditions.py --shape outputs/shape_motion_v2_full_20261005 --output outputs/<new>
       python work/gate_compare/build_three_conditions.py --self-test
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hh_mask_preview import hh_maps

ROOT = Path(__file__).resolve().parents[2]
HH = ROOT / "outputs/gate_compare_20261004/hh_stage2_values.csv"
LABELS = ROOT / "outputs/process_phase_labels_boundary_v5_20261004/phase_labels_boundary_updated.json"
OLD_BUNDLE = ROOT / "outputs/colab_gate_compare_20261004/pipeline_train_bundle_20261004_gatecmp.zip"   # folds.csv
MIN_AREA = 0.1          # same minimum print share as the texture gate threshold, so a recycled frame shows some print
CLOSE, MIN_BED = 25, 0.005   # fill bright bed sparkles; drop bed specks < 0.5% of the crop (dark gaps inside the print)
GRAY = (124, 116, 104)


def selector_keep_mask(crop_rgb):
    """255 = keep. The selector's bed pixel map, cleaned so bed sparkles and dark print gaps don't make holes."""
    bed, _ = hh_maps(crop_rgb)
    bed = cv2.morphologyEx(bed.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE, CLOSE)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bed)
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] < MIN_BED * bed.size:
            bed[lab == j] = 0
    return np.where(bed > 0, 0, 255).astype(np.uint8)


def self_test():
    crop = np.full((1000, 1000, 3), 220, np.uint8)
    crop[:, :300] = (20, 15, 10)                       # dark bed on the left
    crop[:, :300:7] = 230                              # bright sparkle columns inside the bed
    crop[500:504, 600:640] = 30                        # small dark gap inside the print
    keep = selector_keep_mask(crop)
    assert (keep[:, :280] == 0).mean() > .97, "bed (with sparkles) must be masked"
    assert (keep[:, 330:] == 255).all(), "print incl. its small dark gap must be kept"
    print("self-test OK: bed sparkles masked / small print gap kept")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return
    if a.output.exists():
        ap.error("give a NEW --output folder")
    src = {(f["class_name"], f["frame"]): f["source_path"] for f in json.load(open(LABELS, encoding="utf-8"))["frames"]}
    hh = list(csv.DictReader(open(HH, encoding="utf-8-sig")))
    shape = {(r["class_name"], r["frame"]): r for r in json.load(open(a.shape / "tracking_results.json", encoding="utf-8"))["records"]}
    (a.output / "masks/selector").mkdir(parents=True)
    (a.output / "masks/shape").mkdir(parents=True)
    usable = [r for r in hh if r["crop_decision"] == "usable"]
    held = [r for r in hh if r["crop_decision"] != "usable"]
    cond2 = []
    for i, r in enumerate(usable):
        bgr = cv2.imdecode(np.fromfile(src[(r["label"], int(r["frame"]))], np.uint8), cv2.IMREAD_COLOR)
        keep = selector_keep_mask(cv2.cvtColor(bgr[0:1000, 250:1250], cv2.COLOR_BGR2RGB))
        name = f"masks/selector/{r['label']}_{int(r['frame']):08d}.png"
        Image.fromarray(keep).save(a.output / name, optimize=True)
        cond2.append(dict(source=r["source"], label=r["label"], mask=name, masked_share=round(float((keep == 0).mean()), 4)))
        if i % 500 == 0:
            print(f"selector masks {i}/{len(usable)}", flush=True)
    recycled, held_status = [], {}
    for r in held:
        s = shape.get((r["label"], int(r["frame"])))
        status = "missing" if s is None else s["status"]
        ok = s is not None and status == "SUPPORTED" and s["model_visible_area"] >= MIN_AREA
        key = (r["label"], "SUPPORTED_small" if status == "SUPPORTED" and not ok else status)
        held_status[key] = held_status.get(key, 0) + 1
        if ok:
            name = f"masks/shape/{r['label']}_{int(r['frame']):08d}.png"
            m = np.asarray(Image.open(a.shape / "masks" / f"{r['label']}_{int(r['frame']):08d}.png").convert("L"))
            assert m.shape == (1000, 1000)
            Image.fromarray(np.where(m > 127, 255, 0).astype(np.uint8)).save(a.output / name, optimize=True)
            recycled.append(dict(source=r["source"], label=r["label"], mask=name, masked_share=round(float((m <= 127).mean()), 4)))
    write = lambda name, rows, cols: csv.DictWriter(open(a.output / name, "w", encoding="utf-8-sig", newline=""), cols, extrasaction="ignore")
    for name, rows, cols in (("keep_cond1.csv", usable, ["source", "label"]), ("masks_cond2.csv", cond2, ["source", "label", "mask", "masked_share"]),
                             ("keep_cond3.csv", usable + recycled, ["source", "label"]),
                             ("masks_cond3.csv", cond2 + recycled, ["source", "label", "mask", "masked_share"])):
        w = write(name, rows, cols); w.writeheader(); w.writerows(rows)
    count = lambda rows: {c: sum(r["label"] == c for r in rows) for c in ("normal", "underfill_50")}
    summary = dict(cond1_frames=count(usable), cond2_frames=count(usable),
                   cond2_frames_with_any_bed_masked={c: sum(r["label"] == c and r["masked_share"] > 0 for r in cond2) for c in ("normal", "underfill_50")},
                   cond3_frames=count(usable + recycled), cond3_recycled=count(recycled),
                   held_by_shape_status={f"{k[0]}/{k[1]}": v for k, v in sorted(held_status.items())},
                   settings=dict(selector_csv=str(HH), shape_run=str(a.shape), min_area=MIN_AREA, close_kernel=CLOSE,
                                 min_bed_component=MIN_BED, crop="rows 0-1000, cols 250-1250", mask_png="255 = keep"))
    (a.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    # review board: masked inputs as the model will see them (bed-heavy selector frames, evenly spaced recycled frames)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 14)
    pick = sorted(cond2, key=lambda r: -r["masked_share"])[:200:20] + [recycled[i] for i in np.linspace(0, len(recycled) - 1, 20).astype(int)]
    S = 240
    board = Image.new("RGB", (6 * (S + 4), 30 + 5 * (S + 20)), (245, 245, 245))
    d = ImageDraw.Draw(board)
    d.text((6, 6), "모델이 볼 입력: 1~2줄 조건2(선별기 베드 지도) / 3~5줄 조건3 재활용(형상 v2)", font=font, fill="black")
    for i, r in enumerate(pick):
        lab, fr = r["label"], int(r["mask"].split("_")[-1][:8])
        bgr = cv2.imdecode(np.fromfile(src[(lab, fr)], np.uint8), cv2.IMREAD_COLOR)
        crop = cv2.cvtColor(bgr[0:1000, 250:1250], cv2.COLOR_BGR2RGB)
        crop[np.asarray(Image.open(a.output / r["mask"])) == 0] = GRAY
        x, y = (i % 6) * (S + 4), 30 + (i // 6) * (S + 20)
        board.paste(Image.fromarray(crop).resize((S, S)), (x, y))
        d.text((x + 2, y + S + 2), f"{lab[:9]} #{fr} 회색 {r['masked_share']:.2f}", font=font, fill="black")
    board.save(a.output / "review_inputs.jpg", quality=85)
    with zipfile.ZipFile(OLD_BUNDLE) as old, zipfile.ZipFile(a.output / "three_conditions_bundle_20261005.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("folds.csv", old.read("folds.csv"))
        for f in ("train_pilot_classifier.py", "filament_gate.py", "filament_gate_check.py", "pipeline.py"):
            z.write(ROOT / "work" / f, f)
        for p in a.output.rglob("*"):
            if p.suffix in (".csv", ".png", ".json"):
                z.write(p, p.relative_to(a.output).as_posix())
    print("bundle", (a.output / "three_conditions_bundle_20261005.zip").stat().st_size / 2**20, "MiB")


if __name__ == "__main__":
    main()
