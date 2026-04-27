"""Per-class IoU + accuracy report for a trained checkpoint.

Runs the model over data/hackathon/hackathon/<split>.txt and prints a table
of (class name, IoU, Acc, support pixels) for both room and icon heads.

Usage:
    python scripts/eval_classwise.py \
        --weights runs_cubi/2026-04-27-22-59-41/model_best_val_loss_var.pkl \
        --split val.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from floortrans.loaders import FloorplanSVG
from floortrans.loaders.augmentations import DictToTensor
from floortrans.losses import UncertaintyLoss
from floortrans.metrics import runningScore
from floortrans.models import get_model


ROOM_NAMES = [
    "Background", "Outdoor", "Wall", "Kitchen", "LivingRoom", "Bedroom",
    "Bath", "Entry", "Railing", "Storage", "Garage", "Room",
]

ICON_NAMES = [
    "Empty", "Window", "Door", "Closet", "Appliance", "Toilet",
    "Sink", "SaunaBench", "Fireplace", "Bathtub", "Chimney",
]


def fmt_table(title, names, ious, accs, supports):
    print(f"\n=== {title} ===")
    print(f"{'#':>3}  {'class':<12}  {'IoU':>8}  {'Acc':>8}  {'pixels':>12}")
    print("-" * 50)
    valid_iou = []
    valid_acc = []
    for i, name in enumerate(names):
        iou = ious.get(str(i), float("nan"))
        acc = accs.get(str(i), float("nan"))
        sup = int(supports[i])
        if not np.isnan(iou):
            valid_iou.append(iou)
        if not np.isnan(acc):
            valid_acc.append(acc)
        print(f"{i:>3}  {name:<12}  {iou:>8.4f}  {acc:>8.4f}  {sup:>12d}")
    print("-" * 50)
    print(f"  mean IoU: {np.mean(valid_iou):.4f}   mean Acc: {np.mean(valid_acc):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to a .pkl checkpoint")
    ap.add_argument("--data-path", default="data/hackathon/hackathon/")
    ap.add_argument("--split", default="val.txt")
    ap.add_argument("--lmdb-folder", default="../cubi_lmdb/")
    ap.add_argument("--arch", default="hg_furukawa_original")
    ap.add_argument("--n-classes", type=int, default=44)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N samples (0 = all)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    input_slice = [21, 12, 11]
    model = get_model(args.arch, 51)
    model.conv4_ = torch.nn.Conv2d(256, args.n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(args.n_classes, args.n_classes, kernel_size=4, stride=4)
    crit = UncertaintyLoss(input_slice=input_slice, cuda=torch.cuda.is_available())

    ck = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    if "criterion_state" in ck:
        crit.load_state_dict(ck["criterion_state"])
    model.to(device).eval()
    print(f"loaded weights from {args.weights} (epoch={ck.get('epoch', '?')})")

    ds = FloorplanSVG(args.data_path, args.split, format="lmdb",
                      augmentations=DictToTensor(), lmdb_folder=args.lmdb_folder)
    if args.limit > 0:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(args.limit, len(ds)))))
    print(f"evaluating {len(ds)} samples from {args.split}")

    loader = DataLoader(ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())

    rooms_score = runningScore(input_slice[1])
    icons_score = runningScore(input_slice[2])
    rooms_support = np.zeros(input_slice[1], dtype=np.int64)
    icons_support = np.zeros(input_slice[2], dtype=np.int64)

    n_eval = 0
    with torch.no_grad():
        for sample in tqdm(loader, ncols=80):
            img = sample["image"].to(device, non_blocking=True)
            lab = sample["label"].to(device, non_blocking=True)
            out = model(img)
            lab = F.interpolate(lab, size=out.shape[2:], mode="bilinear", align_corners=False)

            room_pred = out[0, input_slice[0]:input_slice[0] + input_slice[1]].argmax(0).cpu().numpy()
            room_gt = lab[0, input_slice[0]].long().cpu().numpy()
            rooms_score.update([room_gt], [room_pred])
            for k in range(input_slice[1]):
                rooms_support[k] += int((room_gt == k).sum())

            icon_pred = out[0, input_slice[0] + input_slice[1]:].argmax(0).cpu().numpy()
            icon_gt = lab[0, input_slice[0] + 1].long().cpu().numpy()
            icons_score.update([icon_gt], [icon_pred])
            for k in range(input_slice[2]):
                icons_support[k] += int((icon_gt == k).sum())

            n_eval += 1

    room_overall, room_per = rooms_score.get_scores()
    icon_overall, icon_per = icons_score.get_scores()

    print(f"\nEvaluated {n_eval} samples.")
    print("\n=== ROOMS — overall ===")
    for k, v in room_overall.items():
        print(f"  {k:<12}: {v:.4f}")
    print("\n=== ICONS — overall ===")
    for k, v in icon_overall.items():
        print(f"  {k:<12}: {v:.4f}")

    fmt_table("Rooms — per class", ROOM_NAMES, room_per["Class IoU"], room_per["Class Acc"], rooms_support)
    fmt_table("Icons — per class", ICON_NAMES, icon_per["Class IoU"], icon_per["Class Acc"], icons_support)


if __name__ == "__main__":
    main()
