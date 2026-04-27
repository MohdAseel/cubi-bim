"""One-shot fine-tune + per-class evaluation on the hackathon dataset.

Run from the repo root:
    cd D:/bim-hack/cubi-bim
    python scripts/finetune.py

That's it. The script will:
    1. Fine-tune model_best_val_loss_var.pkl on data/hackathon/hackathon/
       (lr=1e-5, AMP off, 30 epochs, lr halves on val plateau).
    2. Print a per-class IoU + accuracy table for rooms (12) and icons (11)
       using the best checkpoint from the run.
    3. Save outputs to runs_cubi/<timestamp>/.

Watch live:    tensorboard --logdir runs_cubi
Best checkpoint after run:  runs_cubi/<timestamp>/model_best_val_loss_var.pkl
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_training(args, run_dir: Path):
    cmd = [
        sys.executable, str(REPO / "train.py"),
        "--data-path", "data/hackathon/hackathon/",
        "--weights", "model_best_val_loss_var.pkl",
        "--new-hyperparams", "True",
        "--optimizer", "adam-patience",
        "--l-rate", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--image-size", str(args.image_size),
        "--n-epoch", str(args.epochs),
        "--patience", str(args.patience),
        "--disable-amp",
        "--log-path", str(run_dir.parent),
    ]
    if args.plot_samples:
        cmd += ["--plot-samples", "True"]
    print("[finetune] launching:")
    print("    " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO))


def find_latest_run(log_root: Path) -> Path:
    runs = sorted([p for p in log_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not runs:
        raise RuntimeError(f"no training runs found under {log_root}")
    return runs[-1]


def run_eval(checkpoint: Path):
    cmd = [
        sys.executable, str(REPO / "scripts" / "eval_classwise.py"),
        "--weights", str(checkpoint),
        "--split", "val.txt",
    ]
    print("[eval] launching:")
    print("    " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="fine-tune learning rate (default 1e-5; the original 1e-3 caused NaN with AMP)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--plot-samples", action="store_true")
    ap.add_argument("--skip-training", action="store_true",
                    help="don't train, just run eval on the latest run's best checkpoint")
    ap.add_argument("--checkpoint", default=None,
                    help="explicit path to a .pkl to evaluate (skips training and run-folder lookup)")
    args = ap.parse_args()

    log_root = REPO / "runs_cubi"
    log_root.mkdir(exist_ok=True)

    if args.checkpoint:
        run_eval(Path(args.checkpoint))
        return

    if not args.skip_training:
        before = {p.name for p in log_root.iterdir() if p.is_dir()}
        run_training(args, log_root / datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        after = {p.name for p in log_root.iterdir() if p.is_dir()}
        new_runs = sorted(after - before)
        run_dir = log_root / new_runs[-1] if new_runs else find_latest_run(log_root)
    else:
        run_dir = find_latest_run(log_root)

    ckpt = run_dir / "model_best_val_loss_var.pkl"
    if not ckpt.exists():
        # fallback to other checkpoints train.py writes
        for cand in ["model_best_val_loss.pkl", "model_best_val_acc.pkl", "model_last_epoch.pkl"]:
            if (run_dir / cand).exists():
                ckpt = run_dir / cand
                break
    print(f"[finetune] best checkpoint: {ckpt}")
    run_eval(ckpt)


if __name__ == "__main__":
    main()
