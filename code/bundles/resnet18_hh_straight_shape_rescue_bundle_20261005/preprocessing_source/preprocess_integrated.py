"""Stage 1 event flags + stage 2 image usability; no defect-label prediction.

Stage 1 candidates are never blanket rejections. All frames receive stage 2.
Exported images are byte-identical originals; ROI/resize are analysis-only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import re
import platform
from importlib.metadata import version
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError
from analyze_sequence import DynamicSequenceAnalyzer
from preprocessing_utils import spread, previews

HERE = Path(__file__).resolve().parent
LABELS = ("normal", "underfill_50")


def read_config(path):
    c = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if set(c) != {"analysis_roi", "min_surface_ratio", "max_bed_for_usable",
                  "reject_bed_ratio", "min_detail", "max_surface_variation", "preview_samples"}:
        raise ValueError("Unexpected integrated config keys")
    DynamicSequenceAnalyzer(roi=c["analysis_roi"])
    for key in ("min_surface_ratio", "max_bed_for_usable", "reject_bed_ratio", "min_detail", "max_surface_variation"):
        if type(c[key]) not in (int, float) or not np.isfinite(c[key]) or not 0 < c[key] < 1:
            raise ValueError(f"Invalid {key}")
    if not .35 < c["min_surface_ratio"] or not c["max_bed_for_usable"] < c["reject_bed_ratio"]:
        raise ValueError("Usable and reject thresholds must leave a review range")
    if type(c["preview_samples"]) is not int or not 0 <= c["preview_samples"] <= 30:
        raise ValueError("preview_samples must be an integer in [0,30]")
    return c


def image_metrics(image, c):
    box = tuple(round(v * (image.width if i % 2 == 0 else image.height))
                for i, v in enumerate(c["analysis_roi"]))
    if box[2]-box[0] < 32 or box[3]-box[1] < 32:
        return dict(surface_ratio=0., bed_ratio=0., detail_score=0., surface_variation=0., too_small=True)
    im = image.convert("RGB").crop(box)
    im.thumbnail((640, 640), Image.Resampling.LANCZOS)
    gray_image = im.convert("L")
    gray = np.asarray(gray_image, dtype=float)/255
    low = np.asarray(gray_image.filter(ImageFilter.GaussianBlur(1.2)), dtype=float)/255
    rgb = np.asarray(im, dtype=float)/255
    hsv = np.asarray(im.convert("HSV"), dtype=float)
    hue, sat, val = hsv[..., 0]*360/255, hsv[..., 1], hsv[..., 2]
    bed = (val < 70) | ((hue >= 10) & (hue <= 60) & (sat >= 50) &
                       (sat <= 220) & (val >= 40) & (val <= 180))
    neutral = (gray >= 100/255) & ((rgb.max(2)-rgb.min(2))/np.maximum(rgb.max(2), 1e-6) <= .25)
    total, surfaces, details, variations = 0, 0, [], []
    for y in range(0, gray.shape[0]-31, 32):
        for x in range(0, gray.shape[1]-31, 32):
            s = np.s_[y:y+32, x:x+32]
            total += 1
            if neutral[s].mean() < .65 or bed[s].mean() > .20:
                continue
            q25, median, q75 = np.quantile(gray[s], [.25, .5, .75])
            # Robust local spread discourages mixed bed speckles, tolerates narrow gaps.
            # This remains a proxy: irregular defect texture may still be held for review.
            variation = float((q75-q25)/(median+1e-6))
            if variation > .25:
                continue
            surfaces += 1
            variations.append(variation)
            details.append(float(np.abs(gray[s]-low[s]).mean()/(gray[s].mean()+1e-6)))
    return dict(surface_ratio=surfaces/max(total, 1), bed_ratio=float(bed.mean()),
                detail_score=float(np.median(details)) if details else 0.,
                surface_variation=float(np.median(variations)) if variations else 0., too_small=not total)


def decide(metrics, stage1_candidate, c):
    """No label argument. Temporal flag only corroborates limited-surface bed evidence."""
    surface, bed, detail = (metrics[k] for k in ("surface_ratio", "bed_ratio", "detail_score"))
    if metrics["too_small"]:
        return "review", "image_too_small"
    if surface >= c["min_surface_ratio"] and bed <= c["max_bed_for_usable"]:
        if detail < c["min_detail"]:
            return "review", "low_detail_or_contrast"
        # Bright motion-smeared bed can pass area/detail tests. Require most surface
        # tiles to have modest robust contrast, not merely fall below the loose tile cap.
        # This is NOT a universal blur detector; textured real defects go to review,
        # never permanent rejection based on this extra evidence alone.
        if metrics["surface_variation"] > c["max_surface_variation"]:
            return "review", "high_surface_variation_possible_smeared_bed"
        return "usable", "visible_surface"
    if bed >= c["reject_bed_ratio"] and surface < .35:
        return "rejected", "bed_dominant"
    if stage1_candidate and bed >= .35 and surface < .35:
        return "rejected", "event_and_bed_evidence"
    if surface < .15 and bed >= .25:
        return "rejected", "little_surface_with_bed"
    return "review", "ambiguous_surface_or_bed"


def run(args):
    root, output = args.input.resolve(), args.output.resolve()
    c = read_config(args.config)
    labels = tuple(args.labels)
    if len(labels) != 2 or len(set(labels)) != 2 or any(not re.fullmatch(r"[\w-]+", label) for label in labels):
        raise ValueError("Provide two distinct class folder names using letters, digits, underscore or hyphen")
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output must be separate from raw input tree")
    if args.limit_per_class < 0:
        raise ValueError("limit-per-class must be nonnegative")
    # Validate both inputs before creating the output; one recording folder per label.
    for label in labels:
        DynamicSequenceAnalyzer().load_and_sort_files(root/label, args.limit_per_class)
    output.mkdir(parents=True, exist_ok=False)
    marker = output/"INCOMPLETE"
    marker.write_text("Integrated run not completed", encoding="utf-8")
    rows, stage1_counts = [], {}
    for label in labels:
        analyzer = DynamicSequenceAnalyzer(label, roi=c["analysis_roi"])
        df, report = analyzer.run_analysis(root/label, output/"stage1"/label, args.limit_per_class)
        covered = set()
        with (output/"stage1"/label/"candidate_intervals.csv").open(encoding="utf-8-sig", newline="") as f:
            for event in csv.DictReader(f):
                covered.update(range(int(event["start_frame"]), int(event["end_frame"])+1))
        stage1_counts[label] = dict(events=report["candidate_intervals"], frames=len(covered))
        for i, source_row in enumerate(df.itertuples(), 1):
            candidate = source_row.frame in covered
            row = dict(filename=source_row.filename, label=label, frame=int(source_row.frame),
                       stage1_candidate=candidate, surface_ratio=0., bed_ratio=0., detail_score=0., surface_variation=0.,
                       width=0, height=0, sha256="")
            try:
                raw = Path(source_row.path).read_bytes()
                row["sha256"] = hashlib.sha256(raw).hexdigest()
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
                    row.update(width=im.width, height=im.height)
                    metrics = image_metrics(im, c)
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
                row.update(decision="rejected", reason="unreadable")
            else:
                # Rounded only after decision; avoid threshold changes from serialization.
                decision, reason = decide(metrics, candidate, c)
                row.update({k: round(metrics[k], 6) for k in ("surface_ratio", "bed_ratio", "detail_score", "surface_variation")})
                row.update(decision=decision, reason=reason)
            rows.append(row)
            if i % 250 == 0:
                print(f"Stage 2 {label}: {i}/{len(df)}", flush=True)
    accepted = {label: [r for r in rows if r["label"] == label and r["decision"] == "usable"] for label in labels}
    count = min(map(len, accepted.values()))
    selected = {(r["label"], r["filename"]) for group in accepted.values()
                for r in (spread(group, count) if args.balance else group)}
    for row in rows:
        row["selected"] = (row["label"], row["filename"]) in selected
        row["export_reason"] = "selected" if row["selected"] else (
            "balance_only" if row["decision"] == "usable" else row["reason"])
        if row["selected"] and not args.dry_run:
            raw = (root/row["label"]/row["filename"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise RuntimeError("Source changed during run")
            target = output/"usable"/row["label"]/row["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write(raw)
            if hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]:
                raise RuntimeError("Export hash mismatch")
    with (output/"manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(method="dynamic_events_and_surface_tiles_v2", status="provisional_not_accuracy_validated",
        created_utc=datetime.now(timezone.utc).isoformat(), input=str(root), total=len(rows), config=c,
        balance=args.balance, dry_run=args.dry_run, limit_per_class=args.limit_per_class,
        exported_transform="none; byte-identical originals", classes={}, labels=labels,
        environment=dict(python=platform.python_version(), system=platform.system(),
            libraries={name: version(name) for name in ("numpy", "Pillow", "pandas", "scipy", "matplotlib")}),
        code_sha256={p: hashlib.sha256((HERE/p).read_bytes()).hexdigest()
                     for p in ("preprocess_integrated.py", "analyze_sequence.py", "preprocessing_utils.py")})
    for label in labels:
        group = [r for r in rows if r["label"] == label]
        stats = Counter(r["decision"] for r in group)
        summary["classes"][label] = dict(total=len(group), stage1=stage1_counts[label],
            **{k: stats[k] for k in ("usable", "rejected", "review")},
            selected=sum(r["selected"] for r in group), copied=0 if args.dry_run else sum(r["selected"] for r in group),
            balance_only=sum(r["export_reason"] == "balance_only" for r in group),
            smeared_bed_review=sum(r["reason"] == "high_surface_variation_possible_smeared_bed" for r in group),
            candidate_usable=sum(r["stage1_candidate"] and r["decision"] == "usable" for r in group),
            outside_candidate_rejected=sum(not r["stage1_candidate"] and r["decision"] == "rejected" for r in group))
    (output/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    previews(root, output, rows, c)
    # Explicitly inspect rescued event frames as well as decisions across the full timeline.
    rescued = [r for r in rows if r["stage1_candidate"] and r["decision"] == "usable"]
    if rescued:
        audit = output/"candidate_retained"
        audit.mkdir()
        previews(root, audit, rescued, c)
    blurred_bed = [r for r in rows if r["reason"] == "high_surface_variation_possible_smeared_bed"]
    if blurred_bed:
        audit = output/"smeared_bed_review"
        audit.mkdir()
        previews(root, audit, blurred_bed, c)
    marker.unlink()
    print(json.dumps(summary["classes"], indent=2))
    if args.balance and count == 0:
        print("WARNING: balanced selection is empty because a class has no usable frames")
    print(f"Results: {output}")
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE/"integrated_config.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--labels", nargs=2, default=LABELS, metavar=("CLASS1", "CLASS2"),
                        help="Two class folder names; default: normal underfill_50")
    parser.add_argument("--limit-per-class", type=int, default=0, help="First N consecutive frames; 0=all")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
