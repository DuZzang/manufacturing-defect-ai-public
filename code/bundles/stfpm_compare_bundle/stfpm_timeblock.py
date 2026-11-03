"""Run the teammate STFPM project (unmodified) on a given split and score every test image.

Split CSV columns: fold, role (train/val/buffer), source (e.g. normal\\xxx.jpg), label, label_id, frame_index.
STFPM is unsupervised, so per fold it trains on normal role=train only, then scores normal+underfill role=val.
Same folds.csv as the supervised runs -> the two methods are compared on identical images.

Per fold, the project's own train.py is run as-is (its configs are copied and only paths are changed),
and its own Printer3DDefectPredictor scores the images, so preprocessing and threshold are the teammate's.
A fold whose scores.csv already exists is skipped, so a rerun resumes after a disconnect.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import yaml


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


def link_all(rows, folder, orig_root):
    os.makedirs(folder, exist_ok=True)
    for r in rows:
        src = os.path.join(orig_root, r["source"].replace("\\", "/"))
        dst = os.path.join(folder, os.path.basename(src))
        if not os.path.exists(dst):
            os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="teammate project root (has src/, configs/)")
    ap.add_argument("--orig-root", required=True, help="folder with normal/ and underfill_50/ original photos")
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--out", required=True, help="results folder (Drive); one subfolder per fold")
    ap.add_argument("--work", default="/content/stfpm_work", help="scratch for symlinked data and anomalib logs")
    ap.add_argument("--model-config", default="configs/stfpm_resnet18.yaml", help="relative to --project")
    args = ap.parse_args()

    split = list(csv.DictReader(open(args.split_csv, encoding="utf-8-sig")))
    folds = sorted({int(r["fold"]) for r in split})
    os.makedirs(args.out, exist_ok=True)
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
        data = os.path.join(args.work, f"fold{k}", "data")
        link_all(train, os.path.join(data, "train", "good"), args.orig_root)
        link_all([r for r in test if r["label"] == "normal"], os.path.join(data, "test", "good"), args.orig_root)
        link_all([r for r in test if r["label"] != "normal"], os.path.join(data, "test", "defect"), args.orig_root)

        # teammate configs, paths replaced only
        cfg_dir = os.path.join(fold_out, "configs")
        os.makedirs(cfg_dir, exist_ok=True)
        load = lambda rel: yaml.safe_load(open(os.path.join(args.project, rel), encoding="utf-8"))
        base = load("configs/base.yaml")
        base.update(output_root=os.path.join(args.work, f"fold{k}", "anomalib"),
                    checkpoints_dir=os.path.join(fold_out, "checkpoints"),
                    logs_dir=os.path.join(args.work, f"fold{k}", "logs"))
        dataset = load("configs/printer3d.yaml")
        dataset.update(root=data, normal_dir="train/good", normal_test_dir="test/good", abnormal_dir="test/defect")
        model = load(args.model_config)
        paths = {}
        for name, cfg in (("base", base), ("printer3d", dataset), ("model", model)):
            paths[name] = os.path.join(cfg_dir, f"{name}.yaml")
            yaml.safe_dump(cfg, open(paths[name], "w", encoding="utf-8"), allow_unicode=True)

        print(f"\n===== fold {k}: train normal {len(train)}, test normal "
              f"{sum(r['label'] == 'normal' for r in test)}, test underfill {sum(r['label'] != 'normal' for r in test)}",
              flush=True)
        subprocess.run([sys.executable, "src/train.py", "--config-base", paths["base"],
                        "--config-dataset", paths["printer3d"], "--config-model", paths["model"]],
                       cwd=args.project, check=True)

        # score with the teammate predictor (same preprocessing + saved threshold)
        sys.path.insert(0, args.project)
        from src.models.predictor import Printer3DDefectPredictor
        predictor = Printer3DDefectPredictor(checkpoint_path=os.path.join(fold_out, "checkpoints", "stfpm_model.ckpt"))
        files = [os.path.join(args.orig_root, r["source"].replace("\\", "/")) for r in test]
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
