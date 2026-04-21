"""
download_dataset.py
-------------------
Downloads the CubiCasa5K dataset from Kaggle and extracts it into data/cubicasa5k/.

Prerequisites:
    pip install kaggle
    Place your Kaggle API token at: C:/Users/<you>/.kaggle/kaggle.json
    (Download it from https://www.kaggle.com/settings → API → Create New Token)

Usage:
    python download_dataset.py
    python download_dataset.py --keep-zip
    python download_dataset.py --output-dir data/
"""

import os
import sys
import zipfile
import argparse
import shutil
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────
KAGGLE_DATASET = "qmarva/cubicasa5k"        # owner/dataset-slug
ZIP_NAME       = "cubicasa5k.zip"
DEFAULT_OUTPUT = Path("data")               # extracts to data/cubicasa5k/


# ── Helpers ────────────────────────────────────────────────────────────────────
def ensure_kaggle_token():
    """Check that kaggle.json exists and is readable before doing anything."""
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    if not token_path.exists():
        print("[✗] Kaggle API token not found.")
        print()
        print("  To fix this:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' → click 'Create New Token'")
        print("  3. Save the downloaded kaggle.json to:")
        print(f"       {token_path}")
        print()
        sys.exit(1)

    # Kaggle library requires the file to be chmod 600 on Linux/Mac,
    # but on Windows that check is skipped automatically.
    print(f"[✓] Kaggle token found at {token_path}")


def ensure_kaggle_installed():
    """Import kaggle, offering install guidance if missing."""
    try:
        import kaggle          # noqa: F401  (just checking it imports)
        print("[✓] kaggle package is available")
        return True
    except ImportError:
        print("[✗] The 'kaggle' package is not installed.")
        print("    Run:  pip install kaggle")
        print("    Then re-run this script.")
        sys.exit(1)


# ── Download via Kaggle API ────────────────────────────────────────────────────
def download_from_kaggle(dataset: str, dest_dir: Path):
    """
    Uses the official Kaggle Python API to download and unzip the dataset.
    Equivalent to:  kaggle datasets download -d <dataset> -p <dest_dir> --unzip
    """
    import kaggle  # imported here after the check above

    print(f"[↓] Downloading '{dataset}' from Kaggle ...")
    print(f"    Destination: {dest_dir.resolve()}\n")

    # kaggle.api.dataset_download_files() streams with its own progress bar
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset=dataset,
        path=str(dest_dir),
        unzip=True,       # extract immediately; avoids a second extraction step
        quiet=False,      # show Kaggle's built-in progress bar
    )
    print()


# ── Verify expected output ─────────────────────────────────────────────────────
def find_dataset_root(base: Path) -> Path:
    """
    The Kaggle zip may extract as  base/cubicasa5k/  or directly into base/.
    Walk one level deep to find where train.txt lives.
    """
    for candidate in [base, base / "cubicasa5k"]:
        if (candidate / "train.txt").exists():
            return candidate

    # Fallback: scan one level
    for child in base.iterdir():
        if child.is_dir() and (child / "train.txt").exists():
            return child

    return base   # best guess


def verify(dataset_dir: Path):
    expected = ["train.txt", "val.txt", "test.txt"]
    missing  = [f for f in expected if not (dataset_dir / f).exists()]

    if missing:
        print(f"[!] WARNING: Could not find these files in {dataset_dir}:")
        for f in missing:
            print(f"       — {f}")
        print()
        print("    The extracted layout may differ from what create_lmdb.py expects.")
        print("    Pass --data-path to create_lmdb.py if the folder is named differently.")
    else:
        print("[✓] Dataset structure verified:")
        for f in expected:
            print(f"       ✔  {dataset_dir / f}")
    print()
    return len(missing) == 0


# ── Relocate if extracted into a subfolder ────────────────────────────────────
def normalise_layout(output_dir: Path):
    """
    Ensure the final layout is always  data/cubicasa5k/{train,val,test}.txt
    regardless of how Kaggle extracted the zip.
    """
    target = output_dir / "cubicasa5k"
    actual = find_dataset_root(output_dir)

    if actual == target:
        return  # already correct

    if actual != output_dir and actual.is_dir():
        # e.g. extracted as data/cubicasa5k-something/ → rename
        print(f"[→] Renaming {actual.name}/ → cubicasa5k/ ...")
        actual.rename(target)
    elif (output_dir / "train.txt").exists():
        # extracted flat into data/ → move into cubicasa5k/
        print(f"[→] Moving flat files into cubicasa5k/ ...")
        target.mkdir(exist_ok=True)
        for item in list(output_dir.iterdir()):
            if item.name != "cubicasa5k":
                shutil.move(str(item), str(target / item.name))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download the CubiCasa5K dataset from Kaggle."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"Extraction destination (default: {DEFAULT_OUTPUT}/)"
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="Keep the downloaded zip file (default: Kaggle API unzips and removes it)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  CubiCasa5K — Kaggle Dataset Downloader")
    print("=" * 60)
    print()

    ensure_kaggle_installed()
    ensure_kaggle_token()
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = args.output_dir / "cubicasa5k"

    # Skip if already extracted
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"[→] {dataset_dir} already contains files.")
        print("    Delete the folder and re-run to force a fresh download.\n")
    else:
        download_from_kaggle(KAGGLE_DATASET, args.output_dir)
        normalise_layout(args.output_dir)

    ok = verify(args.output_dir / "cubicasa5k")

    print("=" * 60)
    if ok:
        print("  Dataset ready! Next steps:\n")
        print("  1. Build LMDB databases (needed before training):")
        print("       python create_lmdb.py --txt val.txt")
        print("       python create_lmdb.py --txt test.txt")
        print("       python create_lmdb.py --txt train.txt")
        print()
        print("  2. Train:")
        print("       python train.py")
        print()
        print("  3. Evaluate (requires pre-trained weights):")
        print("       python eval.py --weights model_best_val_loss_var.pkl")
    else:
        print("  Download finished but dataset layout needs manual inspection.")
        print(f"  Check the contents of:  {args.output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
