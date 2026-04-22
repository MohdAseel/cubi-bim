"""
inference.py
============
Backend entry-point for the CubiCasa floorplan raster-to-vector model.

Usage
-----
    from inference import load_model, predict_floorplan

    model = load_model("model_best_val_loss_var.pkl")

    result = predict_floorplan(model, image)
    # result is a dict with keys:
    #   'polygons'      – np.ndarray  (N, 4, 2) wall / icon / opening polygons
    #   'types'         – list of dicts  {type, class, [prob]}
    #   'room_polygons' – list of shapely Polygon objects
    #   'room_types'    – list of dicts  {type, class}
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform as sk_transform
from PIL import Image

from floortrans.models import get_model
from floortrans.loaders.augmentations import RotateNTurns
from floortrans.post_prosessing import split_prediction, get_polygons

# ── Device selection ──────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model architecture constants (must match the checkpoint) ──────────────────
N_CLASSES = 44          # total output channels
SPLIT = [21, 12, 11]   # heatmaps | rooms | icons

# ── Image preprocessing constants (match CubiCasa5K training pipeline) ────────
# The loader resizes the *longest* side to 512 and pads to make it square.
TARGET_SIZE = 512

# ── Post-processing constants ─────────────────────────────────────────────────
HEATMAP_THRESHOLD  = 0.4
ALL_OPENING_TYPES  = [1, 2]  # 1 = window, 2 = door


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(weights_path: str, device: torch.device = DEVICE) -> torch.nn.Module:
    """
    Load the Furukawa hourglass model from a checkpoint file.

    Parameters
    ----------
    weights_path : str
        Path to the ``.pkl`` / ``.pth`` checkpoint produced during training.
    device : torch.device
        Target device (cpu / cuda).

    Returns
    -------
    torch.nn.Module  (eval mode, on *device*)
    """
    model = get_model("hg_furukawa_original", 51)
    model.conv4_ = torch.nn.Conv2d(256, N_CLASSES, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(
        N_CLASSES, N_CLASSES, kernel_size=4, stride=4
    )
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(device)
    return model


# --------------------------------------------------------------------------- #
# Preprocessing  (mirrors samples.ipynb + FloorplanSVG loader logic)
# --------------------------------------------------------------------------- #
def _preprocess(image: Image.Image | np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Convert a raw image into a normalised batch tensor.

    Steps (matching the notebook / FloorplanSVG original_size=True path):
      1. Convert to RGB numpy array.
      2. Scale the *longer* side to TARGET_SIZE; keep aspect ratio.
      3. Pad the shorter side with zeros to make it square (TARGET_SIZE × TARGET_SIZE).
      4. Normalise to [0, 1] and rearrange to CHW.
      5. Return as a (1, 3, H, W) float32 tensor and the *original* (H, W).

    Parameters
    ----------
    image : PIL.Image or np.ndarray (H, W, 3) uint8 RGB

    Returns
    -------
    tensor         : torch.Tensor  shape (1, 3, TARGET_SIZE, TARGET_SIZE)
    original_size  : (int, int)  (height, width) before resizing
    """
    # ── 1. Normalise input to numpy RGB uint8 ─────────────────────────────────
    if isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"), dtype=np.float32)
    else:
        img = np.array(image, dtype=np.float32)
        if img.ndim == 2:                        # greyscale → RGB
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:                  # RGBA → RGB
            img = img[:, :, :3]

    original_h, original_w = img.shape[:2]

    # ── 2. Resize longest side to TARGET_SIZE ────────────────────────────────
    if original_h > original_w:
        new_h = TARGET_SIZE
        new_w = int(round(original_w * TARGET_SIZE / original_h))
    else:
        new_w = TARGET_SIZE
        new_h = int(round(original_h * TARGET_SIZE / original_w))

    img_resized = sk_transform.resize(
        img, (new_h, new_w), order=1, mode="constant",
        anti_aliasing=True, preserve_range=True
    ).astype(np.float32)

    # ── 3. Pad to square ─────────────────────────────────────────────────────
    pad_h = TARGET_SIZE - new_h
    pad_w = TARGET_SIZE - new_w
    img_padded = np.pad(
        img_resized,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # ── 4. Normalise [0, 255] → [0, 1] and convert HWC → CHW ─────────────────
    img_norm = img_padded / 255.0
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()   # (3, H, W)
    tensor = tensor.unsqueeze(0)                                       # (1, 3, H, W)

    return tensor, (original_h, original_w)


# --------------------------------------------------------------------------- #
# 4-Rotation Test-Time Augmentation (TTA)
# --------------------------------------------------------------------------- #
def _run_tta(model: torch.nn.Module,
             tensor: torch.Tensor,
             img_size: tuple[int, int],
             device: torch.device) -> torch.Tensor:
    """
    Run 4-rotation TTA exactly as done in samples.ipynb / metrics.py.

    Returns the averaged prediction tensor shape (1, N_CLASSES, H, W).
    """
    rot = RotateNTurns()
    rotations   = [(0, 0), (1, -1), (2, 2), (-1, 1)]
    height, width = img_size
    pred_count  = len(rotations)
    prediction  = torch.zeros([pred_count, N_CLASSES, height, width])

    tensor = tensor.to(device)
    with torch.no_grad():
        for i, (forward, back) in enumerate(rotations):
            rot_image = rot(tensor, "tensor", forward)
            pred      = model(rot_image)
            pred      = rot(pred, "tensor", back)
            pred      = rot(pred, "points", back)
            pred      = F.interpolate(
                pred, size=(height, width),
                mode="bilinear", align_corners=True
            )
            prediction[i] = pred[0]

    return torch.mean(prediction, dim=0, keepdim=True)   # (1, N_CLASSES, H, W)


# --------------------------------------------------------------------------- #
# Main public function
# --------------------------------------------------------------------------- #
def predict_floorplan(
    model: torch.nn.Module,
    image: "Image.Image | np.ndarray",
    *,
    heatmap_threshold: float = HEATMAP_THRESHOLD,
    use_tta: bool = True,
    device: torch.device = DEVICE,
) -> dict:
    """
    Full inference pipeline: raw image → polygons.

    Parameters
    ----------
    model : torch.nn.Module
        The pre-loaded, eval-mode Furukawa hourglass model.
    image : PIL.Image or np.ndarray
        Raw floorplan image in any of: RGB PIL, RGBA PIL, greyscale/RGB numpy.
    heatmap_threshold : float
        Confidence threshold for the post-processing heatmap step (default 0.4).
    use_tta : bool
        When True (default), average predictions over 4 rotations — same as
        the original notebook.  Set False for faster, single-pass inference.
    device : torch.device
        The device to run the model on.

    Returns
    -------
    dict with keys
    --------------
    'polygons'      : np.ndarray, shape (N, 4, 2)
                      Wall / icon / opening bounding-box polygons.
                      Coordinates are in the *resized* (512 × 512) space.
    'types'         : list[dict]
                      Per-polygon metadata: {'type': 'wall'|'icon', 'class': int, ...}
    'room_polygons' : list[shapely.geometry.Polygon]
                      Merged room regions.
    'room_types'    : list[dict]
                      Per-room metadata: {'type': 'room', 'class': int}
    """
    # ── Preprocessing ─────────────────────────────────────────────────────────
    tensor, original_size = _preprocess(image)
    img_size = (TARGET_SIZE, TARGET_SIZE)   # we operate in the padded space

    # ── Forward pass ──────────────────────────────────────────────────────────
    if use_tta:
        prediction = _run_tta(model, tensor, img_size, device)
    else:
        tensor = tensor.to(device)
        with torch.no_grad():
            prediction = model(tensor)
        prediction = F.interpolate(
            prediction, size=img_size,
            mode="bilinear", align_corners=True
        )

    # ── Post-processing ───────────────────────────────────────────────────────
    # split_prediction: resizes, softmaxes rooms/icons, returns numpy arrays
    heatmaps, rooms, icons = split_prediction(prediction, img_size, SPLIT)

    # get_polygons: full vector extraction
    polygons, types, room_polygons, room_types = get_polygons(
        (heatmaps, rooms, icons),
        heatmap_threshold,
        ALL_OPENING_TYPES,
    )

    return {
        "polygons":      polygons,       # np.ndarray (N, 4, 2)
        "types":         types,          # list[dict]
        "room_polygons": room_polygons,  # list[shapely.Polygon]
        "room_types":    room_types,     # list[dict]
    }


# --------------------------------------------------------------------------- #
# Quick CLI smoke-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 3:
        print("Usage: python inference.py <weights.pkl> <image_path>")
        sys.exit(1)

    weights_path = sys.argv[1]
    image_path   = sys.argv[2]

    print(f"[INFO] Loading model from {weights_path} on {DEVICE} ...")
    m = load_model(weights_path)

    print(f"[INFO] Running inference on {image_path} ...")
    img = Image.open(image_path)
    result = predict_floorplan(m, img)

    print(f"[INFO] Detected {len(result['polygons'])} polygons "
          f"and {len(result['room_polygons'])} room regions.")
    print("[INFO] Types summary:")
    for t in result["types"][:10]:
        print("  ", t)
    if len(result["types"]) > 10:
        print(f"  ... and {len(result['types']) - 10} more")
