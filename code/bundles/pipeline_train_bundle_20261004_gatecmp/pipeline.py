"""Real-time style judgement pipeline: frame -> fixed crop -> filament gate -> CNN -> last-N majority vote.

Preprocessing is the same computation as training with
  train_pilot_classifier.py --input-column source --fixed-crop 0 250 1000 1250 --img-size 512
(PIL RGB -> uint8 -> crop -> bilinear antialias resize -> /255 -> ImageNet normalize); --self-check proves it.
Gate: --gate otsu = the teammate STFPM gatekeeper (Otsu bright-pixel ratio, --min-filament 0.25);
      --gate texture = teammate texture gate v2 (filament_gate.py, --min-filament 0.1). Below it -> frame held.
--mask-bed (texture only) paints the bed gray before resizing, exactly like train_pilot_classifier --mask-bed.

Inputs (frames are processed in file-name order = capture order):
  --input-dir DIR                          any folder of frames (e.g. live frames on the lab PC)
  --split-csv CSV --fold K --orig-root R   held-out frames of fold K (honest demo with that fold's model)
With --split-csv each class is its own print, so the vote window restarts per class; --concat instead feeds
normal then underfill as one stream and reports how many frames the window needs to switch to "underfill".
"""
import argparse
import csv
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filament_gate_check import filament_ratio  # noqa: E402

ROI = (0, 250, 1000, 1250)  # ymin, xmin, ymax, xmax (same box as the teammate STFPM code)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
LABELS = {"normal": 0, "underfill_50": 1}


def build_model(arch, weights, device):
    if arch == "resnet18":
        m = models.resnet18()
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    else:
        m = models.convnext_tiny()
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
    m.load_state_dict(torch.load(weights, map_location="cpu"))   # model_state.pt saved by train_pilot_classifier
    return m.eval().to(device)


def preprocess(path, size, gate="otsu", mask=False):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    crop = rgb[ROI[0]:ROI[2], ROI[1]:ROI[3]]
    if gate == "texture":
        import filament_gate                                     # needs opencv
        crop, ratio = filament_gate.mask_bed(crop) if mask else (crop, filament_gate.filament_ratio(crop))
    else:
        ratio = filament_ratio(crop)
    t = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)          # uint8 CHW, like TF.pil_to_tensor
    t = TF.resize(t, [size, size], interpolation=v2.InterpolationMode.BILINEAR, antialias=True)
    return t, ratio


def self_check(path, size, gate="otsu", mask=False):
    """The pipeline input must equal what train_pilot_classifier.load_images produced during training."""
    from train_pilot_classifier import load_images
    root, rel = os.path.dirname(path), os.path.basename(path)
    trained, _, ratios = load_images([rel], root, size, ROI, gate=gate, mask=mask)
    ours, ratio = preprocess(path, size, gate, mask)
    assert torch.equal(trained[rel], ours), "pipeline preprocessing differs from training"
    assert ratios[rel] == ratio, "gate value differs from training"
    print("self-check ok: pipeline input == training input (uint8, exact)")


def frames_from_args(a):
    """Yield (stream_name, [(path, label_or_None), ...]) in capture order."""
    if a.input_dir:
        files = sorted(f for f in os.listdir(a.input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        label = os.path.basename(os.path.normpath(a.input_dir))
        yield label, [(os.path.join(a.input_dir, f), LABELS.get(label)) for f in files]
        return
    rows = [r for r in csv.DictReader(open(a.split_csv, encoding="utf-8-sig"))
            if int(r["fold"]) == a.fold and r["role"] == "val"]
    streams = []
    for label in LABELS:
        rs = sorted((r for r in rows if r["label"] == label), key=lambda r: int(r["frame_index"]))
        streams.append((label, [(os.path.join(a.orig_root, r["source"].replace("\\", "/")), LABELS[label]) for r in rs]))
    if a.concat:
        yield "normal->underfill", streams[0][1] + streams[1][1]
    else:
        yield from streams


def run_stream(model, frames, a, device):
    votes = deque(maxlen=a.window)       # last N usable-frame verdicts (held frames are skipped, not counted)
    out = []
    for path, y in frames:
        t0 = time.perf_counter()
        x, ratio = preprocess(path, a.img_size, a.gate, a.mask_bed)
        usable = ratio >= a.min_filament
        prob = None
        if usable:
            with torch.no_grad():
                xin = TF.normalize(x.to(device).float().unsqueeze(0) / 255.0, MEAN, STD)
                prob = F.softmax(model(xin).float(), dim=1)[0, 1].item()
            votes.append(prob >= a.threshold)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        if len(votes) < a.min_votes:
            state = "보류"
        else:
            state = "언더필" if sum(votes) * 2 > len(votes) else "정상"
        out.append({"frame": os.path.basename(path), "label": y, "filament_ratio": round(ratio, 3), "usable": int(usable),
                    "prob_underfill": None if prob is None else round(prob, 4),
                    "frame_pred": None if prob is None else int(prob >= a.threshold), "window_state": state,
                    "ms": round(ms, 1)})
    return out


def summarize(name, rows, concat_switch=None):
    n = len(rows)
    used = [r for r in rows if r["usable"]]
    msg = [f"[{name}] 프레임 {n}장 | 게이트 통과 {len(used)}장 ({len(used) / n:.1%})"]
    if rows[0]["label"] is not None:
        if used:
            acc = sum(r["frame_pred"] == r["label"] for r in used) / len(used)
            msg.append(f"프레임 단위 정확도 {acc:.3f}")
        decided = [r for r in rows if r["window_state"] != "보류"]
        if decided:
            ok = sum((r["window_state"] == "언더필") == (r["label"] == 1) for r in decided) / len(decided)
            msg.append(f"다수결 판정률 {len(decided) / n:.1%}, 다수결 정확도 {ok:.3f}")
    ms = sorted(r["ms"] for r in rows[1:]) or [rows[0]["ms"]]   # first frame includes CUDA warm-up
    msg.append(f"1프레임 처리 중앙값 {ms[len(ms) // 2]:.0f}ms (최대 {ms[-1]:.0f}ms)")
    if concat_switch is not None:
        first_bad = next(i for i, r in enumerate(rows) if r["label"] == 1)
        hit = next((i for i, r in enumerate(rows) if i >= first_bad and r["window_state"] == "언더필"), None)
        msg.append("전환 후 언더필 알람까지 " + (f"{hit - first_bad + 1}프레임" if hit is not None else "알람 없음"))
    print(" | ".join(msg))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="model_state.pt from train_pilot_classifier (finetune)")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0", "convnext_tiny"])
    ap.add_argument("--img-size", type=int, default=512, help="must match training (fixed crop runs used 512)")
    ap.add_argument("--input-dir")
    ap.add_argument("--split-csv")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--orig-root")
    ap.add_argument("--concat", action="store_true", help="feed normal then underfill as one stream")
    ap.add_argument("--min-filament", type=float, default=0.25, help="gate: hold frames below this ratio (otsu 0.25, texture 0.1)")
    ap.add_argument("--gate", choices=["otsu", "texture"], default="otsu", help="must match training")
    ap.add_argument("--mask-bed", action="store_true", help="gray out the bed (texture gate); must match training")
    ap.add_argument("--threshold", type=float, default=0.5, help="underfill if probability >= this")
    ap.add_argument("--window", type=int, default=5, help="majority vote over the last N usable frames")
    ap.add_argument("--min-votes", type=int, default=3, help="hold until this many usable frames are in the window")
    ap.add_argument("--out", help="per-frame CSV")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--self-check", metavar="IMAGE", help="only verify preprocessing == training on one image")
    a = ap.parse_args()

    if a.mask_bed and a.gate != "texture":
        sys.exit("--mask-bed needs --gate texture")
    if a.self_check:
        self_check(a.self_check, a.img_size, a.gate, a.mask_bed)
        return
    if not a.input_dir and not (a.split_csv and a.orig_root):
        sys.exit("give --input-dir, or --split-csv with --orig-root")
    device = torch.device(a.device)
    model = build_model(a.arch, a.weights, device)
    all_rows = []
    for name, frames in frames_from_args(a):
        rows = run_stream(model, frames, a, device)
        summarize(name, rows, concat_switch=a.concat or None)
        all_rows += [{"stream": name, **r} for r in rows]
    if a.out:
        with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        print("saved", a.out)


if __name__ == "__main__":
    main()
