# Adding a New Dataset

This guide covers everything you need to integrate an additional floorplan dataset into the training pipeline.

---

## Table of Contents

1. [Understanding the Pipeline](#1-understanding-the-pipeline)
2. [Label Schema You Must Match](#2-label-schema-you-must-match)
3. [Option A — SVG Format (Recommended)](#3-option-a--svg-format-recommended)
4. [Option B — Custom Labels (No SVG)](#4-option-b--custom-labels-no-svg)
5. [Building the LMDB](#5-building-the-lmdb)
6. [Running Training](#6-running-training)
7. [Verification Checklist](#7-verification-checklist)

---

## 1. Understanding the Pipeline

### Data flow

```
PNG image + SVG annotation
        │
        ▼
  FloorplanSVG (txt mode)
  ┌──────────────────────────────────────┐
  │  house.py parses SVG →               │
  │  • 21-channel heatmap tensor         │
  │  • 12-channel room segmentation      │
  │  • 11-channel icon segmentation      │
  └──────────────────────────────────────┘
        │
        ▼
  create_lmdb.py  →  cubi_lmdb/   (key = folder path, value = pickled dict)
        │
        ▼
  FloorplanSVG (lmdb mode)  →  DataLoader  →  model  →  UncertaintyLoss
```

### What each sample looks like at runtime

```python
sample = {
    'image':    torch.Tensor,  # (3, H, W)  float32, NOT yet normalised in lmdb
    'label':    torch.Tensor,  # (44, H, W) float32
    'heatmaps': dict,          # {joint_name: [(x,y), ...]} keypoints
    'scale':    float,         # coef_width between scaled and original image
    'folder':   str,           # relative path used as lmdb key
}
```

The `label` tensor layout is fixed at **44 channels**:

| Channels | Head | Content |
|----------|------|---------|
| 0–20 | Heatmaps | 21 joint/wall-junction Gaussian maps |
| 21–32 | Rooms | 12-class segmentation |
| 33–43 | Icons | 11-class segmentation |

This is controlled by `input_slice = [21, 12, 11]` in `train.py:84`.

---

## 2. Label Schema You Must Match

### Room classes (12 classes, index 0–11)

| Index | Class | Maps from |
|-------|-------|-----------|
| 0 | Background | default (outside) |
| 1 | Outdoor | Outdoor |
| 2 | Wall | Wall |
| 3 | Kitchen | Kitchen |
| 4 | LivingRoom | LivingRoom, Lounge, Dining, EatingArea |
| 5 | Bedroom | Bedroom |
| 6 | Bath | Bath, Sauna |
| 7 | Entry | Entry, HallWay, DraughtLobby |
| 8 | Railing | Railing |
| 9 | Storage | Closet, Storage, DressingRoom |
| 10 | Garage | Garage, CarPort |
| 11 | Room | everything else |

### Icon classes (11 classes, index 0–10)

| Index | Class | Maps from |
|-------|-------|-----------|
| 0 | Empty | (no icon) |
| 1 | Window | Window |
| 2 | Door | Door |
| 3 | Closet | Closet, CoatCloset, CounterTop, Housing, … |
| 4 | Appliance | ElectricalAppliance, GasStove, WoodStove |
| 5 | Toilet | Toilet, Urinal |
| 6 | Sink | Sink, RoundSink, CornerSink, DoubleSink, WaterTap |
| 7 | Sauna bench | SaunaBenchHigh/Low/Mid |
| 8 | Fireplace | Fireplace, PlaceForFireplace, … |
| 9 | Bathtub | Bathtub, BathtubRound |
| 10 | Chimney | Chimney |

`None`-mapped icons (Shower, BaseCabinet, WallCabinet, etc.) are **silently ignored** — they contribute nothing to the label tensor.

---

## 3. Option A — SVG Format (Recommended)

Use this if your dataset annotations can be expressed as SVG polygons and points using the same attribute schema as CubiCasa5k.

### 3.1 Required folder structure

Each floorplan must live in its own folder:

```
data/your_dataset/
├── train.txt
├── val.txt
├── test.txt          ← optional
└── <split>/
    └── <id>/
        ├── F1_scaled.png      ← RGB image resized for training (typically 512×512)
        ├── F1_original.png    ← full-resolution RGB image
        └── model.svg          ← annotation file
```

`train.txt`, `val.txt`, and `test.txt` are plain text files, one relative path per line:

```
/colorful/101/
/high_quality/204/
/colorful/315/
```

The path is relative to the dataset root passed to `--data-path`. It must start and end with `/`.

### 3.2 SVG annotation schema

The SVG parser (`floortrans/loaders/house.py`) expects:

**Walls** — `<polygon>` elements with a `class` attribute. The polygon coordinates define the wall boundary.

**Rooms** — `<polygon>` elements whose `class` attribute matches one of the room names in `all_rooms` (see `house.py:9`). The polygon is flood-filled onto the segmentation canvas.

**Icons** — `<use>` or `<image>` elements whose `class` attribute matches one of the icon names in `all_icons` (see `house.py:204`). Position and bounding-box determine where the heatmap Gaussian is placed.

If your SVG uses different class names, add a mapping in `house.py`:

```python
# house.py — add to rooms_selected and room_name_map
rooms_selected["YourRoomName"] = 4   # maps to LivingRoom index
room_name_map["YourRoomName"] = "LivingRoom"

# and for icons
icons_selected["YourIconName"] = 2   # maps to Door index
```

### 3.3 Merging split files

If you want to train on both CubiCasa5k and your dataset simultaneously, the split files must point into a common root or you concatenate them. The simplest approach: **share the same `--data-path` root**.

```
data/combined/
├── train.txt          ← cat cubicasa5k/train.txt your_data/train.txt > combined/train.txt
├── val.txt
├── cubicasa5k/        ← symlink or copy
│   └── high_quality_architectural/...
└── your_data/
    └── ...
```

Then:

```bash
# merge splits (Linux/macOS)
cat data/cubicasa5k/cubicasa5k/train.txt data/your_dataset/train.txt > data/combined/train.txt
cat data/cubicasa5k/cubicasa5k/val.txt   data/your_dataset/val.txt   > data/combined/val.txt
```

On Windows (PowerShell):

```powershell
Get-Content data/cubicasa5k/cubicasa5k/train.txt, data/your_dataset/train.txt |
    Set-Content data/combined/train.txt -Encoding utf8
```

---

## 4. Option B — Custom Labels (No SVG)

Use this if your dataset has masks, JSON annotations, or any non-SVG format. You subclass `FloorplanSVG` and override `get_txt` to produce the same sample dict.

### 4.1 Create a custom loader

```python
# floortrans/loaders/custom_loader.py
import torch
import numpy as np
import cv2
from floortrans.loaders.svg_loader import FloorplanSVG


class CustomDataset(FloorplanSVG):
    """
    Swap out get_txt to load labels from your own format.
    Everything else (LMDB, augmentations, DataLoader) stays identical.
    """

    def get_txt(self, index):
        folder = self.data_folder + self.folders[index]

        # --- load image ---
        img = cv2.imread(folder + '/image.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        img = np.moveaxis(img, -1, 0).astype(np.float32)   # (3, H, W)

        # --- build label tensor (44, H, W) ---
        label = np.zeros((44, height, width), dtype=np.float32)

        # channels 0–20: heatmap Gaussians — leave zeros if you have no keypoints
        # channels 21–32: room segmentation (one-hot per pixel)
        room_mask = self._load_room_mask(folder, height, width)  # your code
        for class_idx in range(12):
            label[21 + class_idx] = (room_mask == class_idx).astype(np.float32)

        # channels 33–43: icon segmentation
        icon_mask = self._load_icon_mask(folder, height, width)  # your code
        for class_idx in range(11):
            label[33 + class_idx] = (icon_mask == class_idx).astype(np.float32)

        return {
            'image':    torch.tensor(img),
            'label':    torch.tensor(label),
            'heatmaps': {},        # empty dict is fine if you skip heatmap loss
            'scale':    1.0,
            'folder':   self.folders[index],
        }

    def _load_room_mask(self, folder, h, w):
        # example: load a PNG where pixel value = room class index (0–11)
        mask = cv2.imread(folder + '/room_seg.png', cv2.IMREAD_GRAYSCALE)
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    def _load_icon_mask(self, folder, h, w):
        mask = cv2.imread(folder + '/icon_seg.png', cv2.IMREAD_GRAYSCALE)
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
```

### 4.2 Plug the loader into train.py

In `train.py`, replace:

```python
from floortrans.loaders import FloorplanSVG
```

with:

```python
from floortrans.loaders.custom_loader import CustomDataset as FloorplanSVG
```

Everything else — LMDB building, DataLoader, augmentations — works unchanged.

---

## 5. Building the LMDB

The LMDB is a fast binary cache of all pre-parsed samples. **You must rebuild it whenever you add new data.**

```bash
# Build LMDB for train split
python create_lmdb.py \
    --data-path data/combined/ \
    --txt train.txt \
    --lmdb data/combined/cubi_lmdb/

# Build LMDB for val split
python create_lmdb.py \
    --data-path data/combined/ \
    --txt val.txt \
    --lmdb data/combined/cubi_lmdb/
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-path` | `data/cubicasa5k/cubicasa5k/` | Root folder containing split `.txt` files and subfolders |
| `--txt` | *(required)* | Which split file to process (`train.txt`, `val.txt`) |
| `--lmdb` | `data/cubicasa5k/cubi_lmdb/` | Output LMDB directory |
| `--overwrite` | `False` | Re-process entries that already exist in the LMDB |
| `--test` | `False` | Process only first 100 images (smoke test) |

**Incremental updates:** by default `create_lmdb.py` skips keys that already exist. So to add new samples to an existing LMDB without re-processing everything, just run it again without `--overwrite`.

**LMDB size:** the default `map_size` is 200 GB (virtual address space, not actual disk). Increase it in `create_lmdb.py:23` if you have a very large dataset.

---

## 6. Running Training

Point `--data-path` and the `lmdb_folder` inside `train.py` at your combined dataset:

```bash
python train.py \
    --data-path data/combined/ \
    --arch hg_furukawa_original \
    --n-classes 44 \
    --batch-size 8 \
    --image-size 256 \
    --n-epoch 500 \
    --optimizer adam-patience \
    --patience 10 \
    --log-path runs_cubi/
```

The `lmdb_folder` argument in `train.py:57` defaults to `'../cubi_lmdb/'` relative to `data-path`. If your LMDB is at `data/combined/cubi_lmdb/`, change those two lines:

```python
# train.py:56-59
train_set = FloorplanSVG(args.data_path, 'train.txt', format='lmdb',
                         augmentations=aug, lmdb_folder='cubi_lmdb/')
val_set   = FloorplanSVG(args.data_path, 'val.txt',   format='lmdb',
                         augmentations=DictToTensor(), lmdb_folder='cubi_lmdb/')
```

### Quick smoke test

Add `--test` to process only the first 100 samples before committing to a full run:

```bash
python train.py --data-path data/combined/ --test
```

---

## 7. Verification Checklist

Before a full training run, confirm:

- [ ] Every path in `train.txt` / `val.txt` has a corresponding folder under `--data-path`
- [ ] Each folder contains `F1_scaled.png` (or your equivalent image) and `model.svg` (or your loader handles it)
- [ ] LMDB built successfully for both `train.txt` and `val.txt` with no errors in the log
- [ ] A sample loads correctly:
  ```python
  from floortrans.loaders import FloorplanSVG
  ds = FloorplanSVG('data/combined/', 'train.txt', format='lmdb', lmdb_folder='cubi_lmdb/')
  s = ds[0]
  assert s['image'].shape[0] == 3
  assert s['label'].shape[0] == 44
  print(s['image'].shape, s['label'].shape)   # e.g. (3, 512, 512) (44, 512, 512)
  ```
- [ ] Room class indices in your masks are in `[0, 11]`; icon indices in `[0, 10]`
- [ ] No NaN in label tensor: `assert not s['label'].isnan().any()`
- [ ] `--test` training run completes at least one epoch without error
