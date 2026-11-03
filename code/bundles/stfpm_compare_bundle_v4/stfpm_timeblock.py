"""Run the teammate STFPM project (unmodified) on a given split and score every test image.

Split CSV columns: fold, role (train/val/buffer), source (e.g. normal\\xxx.jpg), label, label_id, frame_index.
STFPM is unsupervised, so per fold it trains on normal role=train only, then scores normal+underfill role=val.
Same folds.csv as the supervised runs -> the two methods are compared on identical images.

Per fold, the project's own train.py is run as-is (its configs are copied and only paths are changed),
and its own Printer3DDefectPredictor scores the images, so preprocessing and threshold are the teammate's.
A fold whose scores.csv already exists is skipped, so a rerun resumes after a disconnect.
Scratch data lives in <work>/<name of --out>/foldK, so different runs never share link folders.

--relax copies the project and lifts the GTX-1660 limits hard-coded in its train.py
(backbone resnet18/34 only, batch <= 16, epochs <= 20); the original project is left untouched.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

import yaml

# exact lines in the teammate train.py -> replacement; each must match once
RELAX = [
    ('if backbone not in ("resnet18", "resnet34"):', 'if False:  # relaxed for L4'),
    ('train_batch_size = min(cfg_data.get("train_batch_size", 16), 16)',
     'train_batch_size = cfg_data.get("train_batch_size", 16)  # relaxed for L4'),
    ('eval_batch_size = min(cfg_data.get("eval_batch_size", 16), 16)',
     'eval_batch_size = cfg_data.get("eval_batch_size", 16)  # relaxed for L4'),
    ("if max_epochs > 20:", "if False:  # relaxed for L4 (was: max_epochs > 20)"),
]


def relaxed_copy(project, work):
    dst = os.path.join(work, "project_relaxed")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(project, dst)
    path = os.path.join(dst, "src", "train.py")
    src = open(path, encoding="utf-8").read()
    for old, new in RELAX:
        assert src.count(old) == 1, f"train.py changed, pattern not found exactly once: {old}"
        src = src.replace(old, new)
    open(path, "w", encoding="utf-8").write(src)
    return dst


def auroc(labels, scores):
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def metrics(rows):
    y = [int(r["label_id"]) for r in rows]
    p = [int(r["pred"]) for r in rows]
    tp = sum(a == 1 and b == 1 for a, b in zip(y, p))
    tn = sum(a == 0 and b == 0 for a, b in zip(y, p))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, p))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, p))
    return {"n": len(rows), "accuracy": (tp + tn) / len(rows),
            "underfill_recall": tp / (tp + fn) if tp + fn else None,
            "normal_recall": tn / (tn + fp) if tn + fp else None,
            "auroc": auroc(y, [float(r["score"]) for r in rows]), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def link_all(rows, folder, root, column):
    shutil.rmtree(folder, ignore_errors=True)   # never reuse a folder from an earlier run
    os.makedirs(folder)
    for r in rows:
        src = os.path.join(root, r[column].replace("\\", "/"))
        os.symlink(src, os.path.join(folder, os.path.basename(src)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="teammate project root (has src/, configs/)")
    ap.add_argument("--orig-root", required=True, help="folder that --path-column paths are relative to")
    ap.add_argument("--path-column", default="source",
                    help="split-csv column holding the image path: source (originals) or masked_crop")
    ap.add_argument("--no-roi-crop", action="store_true",
                    help="images are already cropped (e.g. 500x500 masked crops): set roi_crop to None")
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--out", required=True, help="results folder (Drive); one subfolder per fold")
    ap.add_argument("--work", default="/content/stfpm_work", help="scratch for symlinked data and anomalib logs")
    ap.add_argument("--model-config", default="configs/stfpm_resnet18.yaml", help="relative to --project")
    ap.add_argument("--relax", action="store_true", help="lift train.py's GTX-1660 limits in a copy of the project")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    split = list(csv.DictReader(open(args.split_csv, encoding="utf-8-sig")))
    folds = sorted({int(r["fold"]) for r in split})
    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.work, os.path.basename(os.path.normpath(args.out)))
    project = relaxed_copy(args.project, work) if args.relax else args.project
    all_scores = []
    for k in folds:
        fold_out = os.path.join(args.out, f"fold{k}")
        score_csv = os.path.join(fold_out, "scores.csv")
        if os.path.exists(score_csv):
            print(f"fold {k}: scores.csv exists, skipping", flush=True)
            all_scores += list(csv.DictReader(open(score_csv, encoding="utf-8-sig")))
            continue
        rows = [r for r in split if int(r["fold"]) == k]
        train = [r for r in rows if r["role"] == "train" and r["label"] == "normal"]  # unsupervised: normal only
        test = [r for r in rows if r["role"] == "val"]
        scratch = os.path.join(work, f"fold{k}")
        data = os.path.join(scratch, "data")
        col = args.path_column
        link_all(train, os.path.join(data, "train", "good"), args.orig_root, col)
        link_all([r for r in test if r["label"] == "normal"], os.path.join(data, "test", "good"), args.orig_root, col)
        link_all([r for r in test if r["label"] != "normal"], os.path.join(data, "test", "defect"), args.orig_root, col)

        # teammate configs; only paths and the explicitly requested settings change
        cfg_dir = os.path.join(fold_out, "configs")
        os.makedirs(cfg_dir, exist_ok=True)
        load = lambda rel: yaml.safe_load(open(os.path.join(project, rel), encoding="utf-8"))
        base = load("configs/base.yaml")
        base.update(output_root=os.path.join(scratch, "anomalib"),
                    checkpoints_dir=os.path.join(fold_out, "checkpoints"),
                    logs_dir=os.path.join(scratch, "logs"))
        dataset = load("configs/printer3d.yaml")
        dataset.update(root=data, normal_dir="train/good", normal_test_dir="test/good", abnormal_dir="test/defect")
        if args.no_roi_crop:
            dataset["roi_crop"] = None
        if args.batch_size:
            dataset.update(train_batch_size=args.batch_size, eval_batch_size=args.batch_size)
        model = load(args.model_config)
        if args.backbone:
            model["backbone"] = args.backbone
        if args.max_epochs:
            model["max_epochs"] = args.max_epochs
        paths = {}
        for name, cfg in (("base", base), ("printer3d", dataset), ("model", model)):
            paths[name] = os.path.join(cfg_dir, f"{name}.yaml")
            yaml.safe_dump(cfg, open(paths[name], "w", encoding="utf-8"), allow_unicode=True)

        print(f"\n===== fold {k}: train normal {len(train)}, test normal "
              f"{sum(r['label'] == 'normal' for r in test)}, test underfill {sum(r['label'] != 'normal' for r in test)}",
              flush=True)
        # progress bars go to a log file: tens of thousands of lines in the notebook can freeze the Colab tab
        log_path = os.path.join(fold_out, "train_log.txt")
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "src/train.py", "--config-base", paths["base"],
                                   "--config-dataset", paths["printer3d"], "--config-model", paths["model"]],
                                  cwd=project, stdout=log, stderr=subprocess.STDOUT)
        lines = open(log_path, encoding="utf-8", errors="replace").readlines()
        if proc.returncode != 0:
            print("".join(lines[-40:]))
            sys.exit(f"fold {k}: train.py failed, full log: {log_path}")
        print("".join(l for l in lines if "(Threshold)" in l), end="", flush=True)

        # score with the teammate predictor (same preprocessing + saved threshold)
        sys.path.insert(0, project)
        from src.models.predictor import Printer3DDefectPredictor
        predictor = Printer3DDefectPredictor(checkpoint_path=os.path.join(fold_out, "checkpoints", "stfpm_model.ckpt"))
        files = [os.path.join(args.orig_root, r[col].replace("\\", "/")) for r in test]
        results = predictor.predict_batch(files, fail_safe=True, batch_size=16)
        out_rows = []
        for r, res in zip(test, results):
            if res.decision == "ERROR":
                print("  ERROR", r["source"], res.error_message)
                continue
            out_rows.append({"fold": k, "source": r["source"], "label": r["label"], "label_id": r["label_id"],
                             "frame_index": r["frame_index"], "score": res.anomaly_score,
                             "threshold": res.threshold, "pred": int(res.is_anomaly)})
        del predictor
        with open(score_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        m = metrics(out_rows)
        print(f"fold {k}: acc={m['accuracy']:.3f} underfill_recall={m['underfill_recall']:.3f} "
              f"normal_recall={m['normal_recall']:.3f} auroc={m['auroc']:.3f}", flush=True)
        all_scores += out_rows

    per_fold = {k: metrics([r for r in all_scores if int(r["fold"]) == k]) for k in folds}
    # pooled: every test image once; accuracy uses each fold's own unsupervised threshold
    summary = {"pooled": metrics(all_scores), "per_fold": per_fold,
               "fold_mean_auroc": sum(m["auroc"] for m in per_fold.values()) / len(folds),
               "note": "pooled AUROC mixes score scales of different fold models; fold_mean_auroc avoids that"}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
