"""Score an already trained STFPM checkpoint (e.g. the teammate's own model) on the val rows of a split CSV.

Uses the teammate's Printer3DDefectPredictor, so preprocessing and the saved threshold are exactly theirs.
--ckpt-dir must hold stfpm_model.ckpt and model_metadata.json side by side (the predictor reads the
threshold from the json next to the checkpoint).
"""
import argparse
import csv
import os
import sys

from stfpm_timeblock import metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="teammate project root (has src/)")
    ap.add_argument("--ckpt-dir", required=True, help="folder with stfpm_model.ckpt + model_metadata.json")
    ap.add_argument("--split-csv", required=True, help="rows with role=val are scored")
    ap.add_argument("--orig-root", required=True, help="folder that --path-column paths are relative to")
    ap.add_argument("--path-column", default="source")
    ap.add_argument("--out", default=None, help="optional CSV of per-image scores")
    args = ap.parse_args()

    for f in ("stfpm_model.ckpt", "model_metadata.json"):
        if not os.path.exists(os.path.join(args.ckpt_dir, f)):
            sys.exit(f"missing {f} in {args.ckpt_dir}")
    sys.path.insert(0, args.project)
    from src.models.predictor import Printer3DDefectPredictor

    test = [r for r in csv.DictReader(open(args.split_csv, encoding="utf-8-sig")) if r["role"] == "val"]
    predictor = Printer3DDefectPredictor(checkpoint_path=os.path.join(args.ckpt_dir, "stfpm_model.ckpt"))
    files = [os.path.join(args.orig_root, r[args.path_column].replace("\\", "/")) for r in test]
    results = predictor.predict_batch(files, fail_safe=True, batch_size=16)
    rows = [{"source": r["source"], "label": r["label"], "label_id": r["label_id"], "score": x.anomaly_score,
             "threshold": x.threshold, "pred": int(x.is_anomaly)}
            for r, x in zip(test, results) if x.decision != "ERROR"]
    print(f"scored {len(rows)} / {len(test)} images, threshold {predictor.threshold:.6f}")
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in metrics(rows).items()})
    if args.out:
        with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
