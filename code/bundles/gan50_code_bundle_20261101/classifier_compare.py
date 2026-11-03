"""Compare a real-only baseline with GAN and ordinary-image-augmentation additions.

The caller prepares train/validation/test directories *before* running this script.
The same input transformation, negative images, random initialization, optimizer
steps and balanced class sampling are used in each arm. No test or validation score
is used to select a checkpoint: the final optimizer step is saved and evaluated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASSES = ("normal", "underfill")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required image directory not found: {directory}")
    result = sorted((p.resolve() for p in directory.rglob("*")
                     if p.is_file() and p.suffix.lower() in EXTENSIONS),
                    key=lambda p: str(p).casefold())
    if not result:
        raise ValueError(f"No images found in: {directory}")
    return result


def read_image(path: Path, flip: bool = False) -> Image.Image:
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        if flip:
            image = ImageOps.mirror(image)
        return image.copy()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def auc_roc(labels: list[int], scores: list[float]) -> float | None:
    """Rank-sum ROC AUC, accounting for ties; None when only one class is present."""
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        rank_sum += average_rank * sum(label for _, label in ranked[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics_for(rows: list[dict], arm: str, split: str, threshold: float) -> dict:
    tp = sum(r["label"] == 1 and r["prob_underfill"] >= threshold for r in rows)
    tn = sum(r["label"] == 0 and r["prob_underfill"] < threshold for r in rows)
    fp = sum(r["label"] == 0 and r["prob_underfill"] >= threshold for r in rows)
    fn = sum(r["label"] == 1 and r["prob_underfill"] < threshold for r in rows)
    return {"arm": arm, "split": split, "n": len(rows), "threshold": threshold,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": ratio(tp + tn, len(rows)),
            "underfill_recall": ratio(tp, tp + fn),
            "underfill_miss_rate": ratio(fn, tp + fn),
            "normal_false_alarm_rate": ratio(fp, fp + tn),
            "auroc": auc_roc([r["label"] for r in rows],
                             [r["prob_underfill"] for r in rows])}


def predict(model: nn.Module, paths: dict[str, list[Path]], transform, device: torch.device,
            batch_size: int, arm: str, split: str, threshold: float) -> list[dict]:
    model.eval()
    entries = [(path, label) for label, name in enumerate(CLASSES) for path in paths[name]]
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(entries), batch_size):
            batch_entries = entries[offset:offset + batch_size]
            batch = torch.stack([transform(read_image(path)) for path, _ in batch_entries]).to(device)
            scores = torch.sigmoid(model(batch).flatten()).cpu().tolist()
            for (path, label), score in zip(batch_entries, scores):
                rows.append({"arm": arm, "split": split, "path": str(path),
                             "label": label, "label_name": CLASSES[label],
                             "prob_underfill": float(score),
                             "pred_underfill": int(score >= threshold)})
    return rows


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_arm(arm: str, train: dict[str, list[Path]], generated: list[Path],
              transform, args, device: torch.device, output: Path):
    seed_all(args.seed)
    weights = ResNet18_Weights.DEFAULT if args.weights == "imagenet" else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.conv1.parameters(), "lr": args.lr},
        {"params": model.bn1.parameters(), "lr": args.lr},
        {"params": model.layer1.parameters(), "lr": args.lr},
        {"params": model.layer2.parameters(), "lr": args.lr},
        {"params": model.layer3.parameters(), "lr": args.lr},
        {"params": model.layer4.parameters(), "lr": args.lr},
        {"params": model.fc.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    neg_rng = random.Random(args.seed + 7101)
    pos_rng = random.Random(args.seed + 7102)
    choice_rng = random.Random(args.seed + 7103)
    batch_shuffle = random.Random(args.seed + 7104)
    neg_images, pos_images = train["normal"], train["underfill"]
    start = time.perf_counter()
    progress_rows = []
    model.train()
    half = args.batch_size // 2
    for step in range(1, args.steps + 1):
        batch_items = [(neg_images[neg_rng.randrange(len(neg_images))], 0, False)
                       for _ in range(half)]
        for _ in range(half):
            index = pos_rng.randrange(len(pos_images))
            add_extra = choice_rng.random() < 0.5
            if arm == "gan_add" and add_extra:
                batch_items.append((generated[index], 1, False))
            elif arm == "flip_add" and add_extra:
                batch_items.append((pos_images[index], 1, True))
            else:
                batch_items.append((pos_images[index], 1, False))
        batch_shuffle.shuffle(batch_items)
        batch = torch.stack([transform(read_image(path, flip))
                             for path, _, flip in batch_items]).to(device)
        targets = torch.tensor([label for _, label, _ in batch_items],
                               dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch).flatten(), targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss in {arm} at step {step}")
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            progress_rows.append({"arm": arm, "step": step, "loss": float(loss.detach().cpu()),
                                  "elapsed_seconds": round(time.perf_counter() - start, 3)})
            print(f"{arm}: {step}/{args.steps} loss={progress_rows[-1]['loss']:.4f}", flush=True)
    checkpoint = output / f"{arm}_final.pt"
    torch.save({"architecture": "resnet18", "state_dict": model.state_dict(),
                "arm": arm, "seed": args.seed, "steps": args.steps,
                "weights_initialization": args.weights,
                "imagenet_mean": IMAGENET_MEAN, "imagenet_std": IMAGENET_STD,
                "input_size": args.size, "threshold": args.threshold}, checkpoint)
    save_csv(output / f"{arm}_training_log.csv", progress_rows)
    return model, round(time.perf_counter() - start, 3)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True,
                   help="Root with train_real/, val/, test/ and normal/, underfill/ subfolders")
    p.add_argument("--gan-dir", type=Path, required=True, help="Directory with 50 generated PNG/JPG images")
    p.add_argument("--out", type=Path, required=True, help="New output directory (must be empty or absent)")
    p.add_argument("--steps", type=int, default=400, help="Identical optimizer steps for every arm")
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--batch-size", type=int, default=16, help="Even, with half normal / half underfill")
    p.add_argument("--size", type=int, default=256, help="Identical model input side for all images")
    p.add_argument("--normal-max", type=int, default=500, help="Maximum normal training images")
    p.add_argument("--expected-underfill", type=int, default=50,
                   help="Required count of real training underfill and generated images (default 50)")
    p.add_argument("--lr", type=float, default=1e-4, help="Backbone learning rate")
    p.add_argument("--head-lr", type=float, default=1e-3, help="Classifier head learning rate")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Fixed in advance; validation/test do not choose it")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--skip-flip", action="store_true", help="Run two arms only")
    p.add_argument("--weights", choices=("imagenet", "none"), default="imagenet",
                   help="'none' is for synthetic smoke testing, not the intended experiment")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("--steps must be positive and --batch-size must be a positive even number")
    if args.normal_max < 1 or args.expected_underfill < 1 or args.size < 32 or args.log_every < 1:
        raise ValueError("--normal-max, --expected-underfill, --log-every must be positive; --size >= 32")
    if not (0 <= args.threshold <= 1):
        raise ValueError("--threshold must be between 0 and 1")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to mix results into nonempty output directory: {args.out}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. In Colab choose a GPU runtime, or use --device cpu.")
    dataset = {split: {name: images_in(args.data / split / name) for name in CLASSES}
               for split in ("train_real", "val", "test")}
    generated = images_in(args.gan_dir)
    actual_underfill = len(dataset["train_real"]["underfill"])
    if actual_underfill != args.expected_underfill or len(generated) != args.expected_underfill:
        raise ValueError(f"Expected {args.expected_underfill} real underfill and generated images, found "
                         f"{actual_underfill} and {len(generated)}")
    # Exact filename reuse across train/val/test within a class is a clear leakage
    # signal. Different names can still refer to adjacent or duplicate frames;
    # the data preparation stage must verify parent IDs and the time buffer.
    for name in CLASSES:
        by_split = {split: {p.name.casefold() for p in dataset[split][name]}
                    for split in dataset}
        for first, second in (("train_real", "val"), ("train_real", "test"), ("val", "test")):
            overlap = by_split[first] & by_split[second]
            if overlap:
                raise ValueError(f"Same filename appears in {name}/{first} and {name}/{second}: "
                                 f"{sorted(overlap)[:5]}")
    if len(dataset["train_real"]["normal"]) > args.normal_max:
        # The frame split is already fixed; deterministic selection only limits
        # the training normals and does not choose validation/test images.
        rng = random.Random(args.seed + 200)
        dataset["train_real"]["normal"] = sorted(
            rng.sample(dataset["train_real"]["normal"], args.normal_max),
            key=lambda p: str(p).casefold())
    args.out.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    manifest = []
    for split, groups in dataset.items():
        for name, paths in groups.items():
            manifest.extend({"split": split, "label": name, "path": str(p)} for p in paths)
    manifest.extend({"split": "generated_train_only", "label": "underfill", "path": str(p)}
                    for p in generated)
    save_csv(args.out / "image_manifest.csv", manifest)
    arms = ["real_only", "gan_add"] + ([] if args.skip_flip else ["flip_add"])
    settings = {
        "data_root": str(args.data.resolve()), "gan_dir": str(args.gan_dir.resolve()),
        "arms": arms, "seed": args.seed, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "weights": args.weights,
        "architecture": "resnet18", "input_size": args.size, "color": "RGB",
        "resize": "PIL RGB, bilinear, square", "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD, "steps": args.steps,
        "batch_size": args.batch_size, "backbone_lr": args.lr, "head_lr": args.head_lr,
        "weight_decay": args.weight_decay, "threshold": args.threshold,
        "class_sampling": "every batch: 50% normal, 50% underfill; with replacement",
        "positive_sampling": "real_only: 100% real; gan_add/flip_add: 50% real, 50% added",
        "ordinary_augmentation": "deterministic horizontal flip of same underfill training images",
        "checkpoint_policy": "final step fixed in advance; validation/test are not used for selection",
        "counts": {split: {name: len(paths) for name, paths in groups.items()}
                   for split, groups in dataset.items()} | {"generated": len(generated)},
        "caveats": ["one developmental split and one seed; not independent new print evaluation",
                    "parent-frame and time-buffer leakage require separate manifest verification",
                    "matching RGB dimensions does not guarantee same surface scale or masking",
                    "flip validity for this printing process requires human review"],
    }
    (args.out / "run_settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    all_predictions, all_metrics = [], []
    timing = {}
    for arm in arms:
        model, timing[arm] = train_arm(arm, dataset["train_real"], generated, transform, args, device, args.out)
        for split in ("val", "test"):
            rows = predict(model, dataset[split], transform, device, args.batch_size,
                           arm, split, args.threshold)
            all_predictions.extend(rows)
            all_metrics.append(metrics_for(rows, arm, split, args.threshold))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_csv(args.out / "predictions.csv", all_predictions)
    save_csv(args.out / "metrics.csv", all_metrics)
    (args.out / "timings.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
