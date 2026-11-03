"""Pilot binary classifier (normal=0 vs underfill_50=1) on time-block folds.

Reads folds.csv from make_pilot_splits.py, trains an ImageNet-pretrained
torchvision model for every fold and seed, and writes per-run metrics,
out-of-fold predictions, misclassified images, weights and the environment
to a new output folder. Every model option below is shared by --arch, so a
ResNet18 run with the same folds, input and budget is directly comparable.

Scores are pilot numbers from blocks inside one capture run per class; they
do not measure generalisation to new print experiments. Hyperparameters and
epoch counts are fixed in advance and the last epoch is reported (no early
stopping on the validation block).
"""
import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

LABEL_IDS = {"normal": 0, "underfill_50": 1}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODE_DEFAULTS = {
    # linear: backbone frozen (BatchNorm kept in eval mode), only the new head trains
    "linear": {"epochs": 30, "lr_backbone": 0.0, "lr_head": 1e-3},
    # finetune: all layers train, backbone at a 10x lower learning rate than the head
    "finetune": {"epochs": 20, "lr_backbone": 1e-4, "lr_head": 1e-3},
}
AUGMENT_DESC = "train only: RandomHorizontalFlip(p=0.5), ColorJitter(brightness=0.1, contrast=0.1); no blur, no random crop"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds-csv", required=True)
    ap.add_argument("--pilot-root", required=True, help="folder that masked_crop/crop paths are relative to")
    ap.add_argument("--output", required=True, help="new, empty output folder")
    ap.add_argument("--arch", choices=["efficientnet_b0", "resnet18", "convnext_tiny"], required=True)
    ap.add_argument("--modes", nargs="+", choices=list(MODE_DEFAULTS), default=["linear", "finetune"])
    ap.add_argument("--input-column", choices=["masked_crop", "crop", "source"], default="masked_crop",
                    help="source = original photo (pair with --fixed-crop and --pilot-root at the originals folder)")
    ap.add_argument("--fixed-crop", type=int, nargs=4, metavar=("YMIN", "XMIN", "YMAX", "XMAX"), default=None,
                    help="cut this box from every image before resizing, e.g. 0 250 1000 1250 (the STFPM project's crop)")
    ap.add_argument("--img-size", type=int, default=224, help="whole crop is resized to this square; no center crop")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=None, help="override the per-mode epoch count (smoke tests)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="subset of folds (default: all)")
    ap.add_argument("--device", default="auto", help="auto, cuda or cpu")
    ap.add_argument("--no-amp", action="store_true", help="disable float16 autocast on CUDA")
    ap.add_argument("--fast", action="store_true",
                    help="cudnn.benchmark on, deterministic algorithms off: faster, but a rerun with the same seed "
                         "can differ slightly")
    ap.add_argument("--save-weights", choices=["auto", "full", "none"], default="auto",
                    help="auto: head only for linear, full model for finetune")
    ap.add_argument("--torch-home", default=None, help="cache for pretrained weights (default: <this folder>/torch_cache)")
    return ap.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch):
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        head = model.classifier
    elif arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        # classifier = LayerNorm2d -> Flatten -> Linear; only the Linear is the new head
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
        head = model.classifier[2]
    else:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        head = model.fc
    return model, head, f"{weights.__class__.__name__}.{weights.name}"


def load_images(rel_paths, root, size, fixed_crop=None):
    images, sizes = {}, {}
    for rel in rel_paths:
        # CSV paths use Windows "\"; "/" works on both Windows and Linux (Colab)
        with Image.open(os.path.join(root, rel.replace("\\", "/"))) as im:
            sizes[rel] = im.size
            t = TF.pil_to_tensor(im.convert("RGB"))
        if fixed_crop:
            ymin, xmin, ymax, xmax = fixed_crop
            t = t[:, ymin:ymax, xmin:xmax]
        images[rel] = TF.resize(t, [size, size], interpolation=v2.InterpolationMode.BILINEAR, antialias=True)
    return images, sizes


def to_input(batch_uint8, device, augment=None):
    # copy uint8 to the device first (4x less data), then float/augment/normalize there;
    # augment's random draws still come from the CPU generator, so seeds pick the same flips/jitters
    x = batch_uint8.to(device, non_blocking=True).float() / 255.0
    if augment is not None:
        x = torch.stack([augment(img) for img in x])
    return TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)


def auroc(labels, scores):
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def binary_metrics(labels, probs, threshold=0.5):
    preds = [int(p >= threshold) for p in probs]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, preds))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, preds))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, preds))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, preds))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {
        "n": len(labels), "accuracy": (tp + tn) / len(labels) if labels else None,
        "underfill_precision": prec, "underfill_recall": rec, "underfill_f1": f1,
        "normal_recall": spec,
        "balanced_accuracy": (rec + spec) / 2 if rec is not None and spec is not None else None,
        "auroc": auroc(labels, probs), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


@torch.no_grad()
def predict(model, items, images, col, device, batch_size, amp):
    model.eval()
    probs, losses = [], []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        x = to_input(torch.stack([images[it[col]] for it in chunk]), device)
        y = torch.tensor([it["label_id"] for it in chunk], device=device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        losses.append(F.cross_entropy(logits.float(), y, reduction="sum").item())
        probs.extend(F.softmax(logits.float(), dim=1)[:, 1].tolist())
    return probs, sum(losses) / max(len(items), 1)


def run_one(args, mode, seed, fold, train_items, val_items, images, device, amp, run_dir):
    cfg = dict(MODE_DEFAULTS[mode])
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    col = args.input_column
    set_seed(seed * 100 + fold)
    model, head, weights_name = build_model(args.arch)
    model.to(device)

    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    if mode == "linear":
        for p in backbone:
            p.requires_grad_(False)
        groups = [{"params": list(head.parameters()), "lr": cfg["lr_head"]}]
    else:
        groups = [{"params": backbone, "lr": cfg["lr_backbone"]},
                  {"params": list(head.parameters()), "lr": cfg["lr_head"]}]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps = cfg["epochs"] * math.ceil(len(train_items) / args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    augment = v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.ColorJitter(brightness=0.1, contrast=0.1)])
    order_gen = torch.Generator().manual_seed(seed * 1000 + fold)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.time()
        if mode == "linear":
            model.eval()
            head.train()
        else:
            model.train()
        perm = torch.randperm(len(train_items), generator=order_gen).tolist()
        total, count = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            if len(idx) < 2:
                continue
            batch = [train_items[j] for j in idx]
            x = to_input(torch.stack([images[it[col]] for it in batch]), device, augment)
            y = torch.tensor([it["label_id"] for it in batch], device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
            count += len(idx)
        val_probs, val_loss = predict(model, val_items, images, col, device, args.batch_size, amp)
        vm = binary_metrics([it["label_id"] for it in val_items], val_probs)
        history.append({"epoch": epoch, "train_loss": total / max(count, 1), "val_loss": val_loss,
                        "val_accuracy": vm["accuracy"], "val_underfill_recall": vm["underfill_recall"],
                        # epoch 1 includes one-time warm-up (cudnn.benchmark search etc.); later epochs are steady speed
                        "epoch_seconds": round(time.time() - epoch_start, 2)})
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - start
    peak_alloc = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else None

    train_probs, _ = predict(model, train_items, images, col, device, args.batch_size, amp)
    train_m = binary_metrics([it["label_id"] for it in train_items], train_probs)
    val_m = binary_metrics([it["label_id"] for it in val_items], val_probs)

    os.makedirs(run_dir, exist_ok=True)
    write_csv(os.path.join(run_dir, "history.csv"), history)
    preds = [{"source": it["source"], "label": it["label"], "label_id": it["label_id"],
              "frame_index": it["frame_index"], "prob_underfill": round(p, 6), "pred": int(p >= 0.5)}
             for it, p in zip(val_items, val_probs)]
    write_csv(os.path.join(run_dir, "val_predictions.csv"), preds)
    save = args.save_weights
    if save == "full" or (save == "auto" and mode == "finetune"):
        torch.save(model.state_dict(), os.path.join(run_dir, "model_state.pt"))
    elif save == "auto":
        torch.save(head.state_dict(), os.path.join(run_dir, "head_state.pt"))
    result = {"mode": mode, "seed": seed, "fold": fold, "epochs": cfg["epochs"],
              "lr_backbone": cfg["lr_backbone"], "lr_head": cfg["lr_head"],
              "n_train": len(train_items), "n_val": len(val_items),
              "train_accuracy": train_m["accuracy"], "seconds": round(seconds, 2),
              "peak_alloc_mib": round(peak_alloc, 1) if peak_alloc else None,
              "peak_reserved_mib": round(peak_reserved, 1) if peak_reserved else None,
              "weights": weights_name, **{f"val_{k}": v for k, v in val_m.items()}}
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result, preds


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt(v, digits=3):
    return "-" if v is None else f"{v:.{digits}f}"


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return m, s


def environment_text(device):
    lines = [f"python: {sys.version}", f"executable: {sys.executable}", f"platform: {platform.platform()}",
             f"torch: {torch.__version__}", f"torch.version.cuda: {torch.version.cuda}",
             f"cudnn: {torch.backends.cudnn.version()}", f"cuda available: {torch.cuda.is_available()}",
             f"device used: {device}"]
    if torch.cuda.is_available():
        lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
        lines.append(f"gpu capability: {torch.cuda.get_device_capability(0)}")
    try:
        import torchvision
        lines.append(f"torchvision: {torchvision.__version__}")
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        lines.append(f"nvidia-smi: {smi.stdout.strip()}")
    except Exception as e:  # nvidia-smi is optional
        lines.append(f"nvidia-smi: unavailable ({e})")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    lines.append("\n# pip freeze\n" + freeze.stdout)
    return "\n".join(lines)


def main():
    args = parse_args()
    if os.path.exists(args.output) and os.listdir(args.output):
        sys.exit(f"output folder is not empty: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    torch_home = args.torch_home or os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_cache")
    torch.hub.set_dir(os.path.join(torch_home, "hub"))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but torch.cuda.is_available() is False")
    amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = args.fast
    torch.backends.cudnn.deterministic = not args.fast
    torch.use_deterministic_algorithms(not args.fast, warn_only=True)
    print(f"device={device} amp={amp} cuda_available={torch.cuda.is_available()}", flush=True)

    with open(args.folds_csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["fold"] = int(r["fold"])
        r["label_id"] = int(r["label_id"])
        r["frame_index"] = int(r["frame_index"])
        if LABEL_IDS[r["label"]] != r["label_id"]:
            sys.exit(f"label mapping mismatch for {r['source']}")
    folds = sorted({r["fold"] for r in rows}) if args.folds is None else args.folds

    col = args.input_column
    unique = sorted({r[col] for r in rows})
    images, sizes = load_images(unique, args.pilot_root, args.img_size, args.fixed_crop)
    size_counts = defaultdict(int)
    for s in sizes.values():
        size_counts[f"{s[0]}x{s[1]}"] += 1
    print(f"loaded {len(images)} images, original sizes {dict(size_counts)}", flush=True)

    settings = {k: v for k, v in vars(args).items()}
    settings.update({"mode_defaults": MODE_DEFAULTS, "label_ids": LABEL_IDS, "augment": AUGMENT_DESC,
                     "preprocess": f"RGB -> fixed crop {args.fixed_crop} [ymin,xmin,ymax,xmax] -> "
                                   f"resize whole crop to {args.img_size}x{args.img_size} (bilinear, antialias) "
                                   f"-> [0,1] -> ImageNet mean/std; no center crop",
                     "optimizer": "AdamW, cosine decay per step, cross-entropy, last epoch reported",
                     "original_sizes": dict(size_counts), "folds_used": folds, "amp": amp,
                     "torch_home": torch_home})
    with open(os.path.join(args.output, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output, "command.txt"), "w", encoding="utf-8") as f:
        f.write(subprocess.list2cmdline([sys.executable] + sys.argv) + "\n")
    with open(os.path.join(args.output, "environment.txt"), "w", encoding="utf-8") as f:
        f.write(environment_text(device))

    # Shortcut check: can simple non-learned statistics already separate the classes?
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], r)
    labels = [r["label_id"] for r in by_source.values()]
    excl = [float(r["excluded_fraction"]) for r in by_source.values()]
    bright = [images[r[col]].float().mean().item() for r in by_source.values()]
    shortcut = {
        "n": len(labels),
        "auroc_excluded_fraction": auroc(labels, excl),
        "auroc_mean_intensity": auroc(labels, bright),
        "mean_excluded_fraction": {lab: mean_std([e for e, r in zip(excl, by_source.values()) if r["label"] == lab])[0]
                                   for lab in LABEL_IDS},
        "mean_intensity_0_255": {lab: mean_std([b for b, r in zip(bright, by_source.values()) if r["label"] == lab])[0]
                                 for lab in LABEL_IDS},
        "note": "AUROC near 0.5 means the statistic alone does not separate the classes; near 0 or 1 means it does.",
    }
    with open(os.path.join(args.output, "shortcut_check.json"), "w", encoding="utf-8") as f:
        json.dump(shortcut, f, indent=2)

    results, oof = [], []
    for mode in args.modes:
        for seed in args.seeds:
            for fold in folds:
                train_items = [r for r in rows if r["fold"] == fold and r["role"] == "train"]
                val_items = [r for r in rows if r["fold"] == fold and r["role"] == "val"]
                run_dir = os.path.join(args.output, "runs", mode, f"seed{seed}", f"fold{fold}")
                res, preds = run_one(args, mode, seed, fold, train_items, val_items, images, device, amp, run_dir)
                results.append(res)
                oof.extend({"mode": mode, "seed": seed, "fold": fold, **p} for p in preds)
                print(f"{mode} seed={seed} fold={fold} val_acc={fmt(res['val_accuracy'])} "
                      f"underfill_recall={fmt(res['val_underfill_recall'])} auroc={fmt(res['val_auroc'])} "
                      f"train_acc={fmt(res['train_accuracy'])} {res['seconds']}s peak={res['peak_alloc_mib']}MiB",
                      flush=True)

    write_csv(os.path.join(args.output, "fold_metrics.csv"), results)
    write_csv(os.path.join(args.output, "oof_predictions.csv"), oof)

    # Pooled out-of-fold metrics: every image is predicted once per seed when all folds are used.
    pooled = []
    for mode in args.modes:
        for seed in args.seeds:
            sub = [p for p in oof if p["mode"] == mode and p["seed"] == seed]
            m = binary_metrics([p["label_id"] for p in sub], [p["prob_underfill"] for p in sub])
            pooled.append({"mode": mode, "seed": seed, **m})
    write_csv(os.path.join(args.output, "pooled_oof_metrics.csv"), pooled)

    wrong = defaultdict(lambda: {"n_wrong": 0, "n_runs": 0, "probs": []})
    for p in oof:
        key = (p["mode"], p["source"])
        wrong[key]["n_runs"] += 1
        wrong[key]["probs"].append(p["prob_underfill"])
        wrong[key].update(label=p["label"], fold=p["fold"], frame_index=p["frame_index"])
        if p["pred"] != p["label_id"]:
            wrong[key]["n_wrong"] += 1
    mis = [{"mode": k[0], "source": k[1], "label": v["label"], "fold": v["fold"], "frame_index": v["frame_index"],
            "wrong_in_seeds": f"{v['n_wrong']}/{v['n_runs']}",
            "mean_prob_underfill": round(sum(v["probs"]) / len(v["probs"]), 4)}
           for k, v in wrong.items() if v["n_wrong"] > 0]
    mis.sort(key=lambda r: (r["mode"], -int(r["wrong_in_seeds"].split("/")[0]), r["source"]))
    write_csv(os.path.join(args.output, "misclassified.csv"), mis)

    keys = ["accuracy", "underfill_precision", "underfill_recall", "underfill_f1", "normal_recall",
            "balanced_accuracy", "auroc"]
    summary = []
    for mode in args.modes:
        fr = [r for r in results if r["mode"] == mode]
        pr = [r for r in pooled if r["mode"] == mode]
        row = {"mode": mode, "runs": len(fr)}
        for k in keys:
            row[f"fold_mean_{k}"], row[f"fold_std_{k}"] = mean_std([r[f"val_{k}"] for r in fr])
            row[f"pooled_mean_{k}"], _ = mean_std([r[k] for r in pr])
        for c in ("tn", "fp", "fn", "tp"):
            row[f"pooled_{c}_sum_over_seeds"] = sum(r[c] for r in pr)
        row["mean_train_accuracy"], _ = mean_std([r["train_accuracy"] for r in fr])
        row["mean_seconds_per_run"], _ = mean_std([r["seconds"] for r in fr])
        row["max_peak_alloc_mib"] = max((r["peak_alloc_mib"] or 0) for r in fr) or None
        row["max_peak_reserved_mib"] = max((r["peak_reserved_mib"] or 0) for r in fr) or None
        summary.append(row)
    write_csv(os.path.join(args.output, "summary.csv"), summary)
    print(json.dumps({"summary": summary, "shortcut_check": shortcut}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
