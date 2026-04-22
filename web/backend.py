"""FastAPI backend — runs inference and serves the frontend."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from inference import TARGET_SIZE, load_model, predict_floorplan

# ── Label / colour tables ──────────────────────────────────────────────────────
ROOM_LABELS = [
    "Background", "Outdoor", "Wall", "Kitchen", "Living Room",
    "Bedroom", "Bathroom", "Entry/Corridor", "Railing", "Storage",
    "Garage", "Undefined",
]
ROOM_COLORS = [
    "#808080", "#90EE90", "#8B4513", "#FFD700", "#87CEEB",
    "#DDA0DD", "#87CEFA", "#F0E68C", "#C0C0C0", "#D2691E",
    "#708090", "#A9A9A9",
]
ICON_LABELS = [
    "No Icon", "Window", "Door", "Closet", "Electrical Appliance",
    "Toilet", "Sink", "Sauna Bench", "Fireplace", "Bathtub", "Chimney",
]
ICON_COLORS = [
    "#FFFFFF", "#4169E1", "#8B0000", "#9370DB", "#FF8C00",
    "#20B2AA", "#2E8B57", "#CD853F", "#FF4500", "#4682B4", "#696969",
]
WALL_COLOR = "#C0392B"

FRONTEND_DIR = Path(__file__).parent / "frontend"
MODEL_PATH   = Path(__file__).parent.parent / "model_best_val_loss_var.pkl"

app   = FastAPI(title="CubiCasa Floorplan Editor")
model = None   # loaded once on startup


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
async def _load_model() -> None:
    global model
    if MODEL_PATH.exists():
        print(f"Loading model from {MODEL_PATH} …")
        model = load_model(str(MODEL_PATH))
        print("Model ready.")
    else:
        print(f"[WARN] Model not found at {MODEL_PATH}")


def _label_and_color(ptype: str, cls: int) -> tuple[str, str]:
    if ptype == "wall":
        return ROOM_LABELS[cls] if cls < len(ROOM_LABELS) else f"Wall {cls}", WALL_COLOR
    if ptype == "room":
        return (
            ROOM_LABELS[cls] if cls < len(ROOM_LABELS) else f"Room {cls}",
            ROOM_COLORS[cls % len(ROOM_COLORS)],
        )
    # icon / opening
    return (
        ICON_LABELS[cls] if cls < len(ICON_LABELS) else f"Icon {cls}",
        ICON_COLORS[cls % len(ICON_COLORS)],
    )


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    raw  = await image.read()
    pil  = Image.open(io.BytesIO(raw))

    # Store original size so the frontend can compute the correct scale factor
    orig_w, orig_h = pil.size
    long_side = max(orig_w, orig_h)
    resize_factor = TARGET_SIZE / long_side
    resized_w = round(orig_w * resize_factor)
    resized_h = round(orig_h * resize_factor)

    result = predict_floorplan(model, pil)

    elements: list[dict] = []
    for idx, (poly, t) in enumerate(zip(result["polygons"], result["types"])):
        cls   = int(t.get("class", 0))
        ptype = str(t.get("type", ""))
        label, color = _label_and_color(ptype, cls)
        elements.append({
            "id":       idx,
            "vertices": poly.tolist(),   # [[x,y], [x,y], [x,y], [x,y]]
            "type":     ptype,
            "class":    cls,
            "label":    label,
            "color":    color,
        })

    rooms: list[dict] = []
    for idx, (poly, t) in enumerate(zip(result["room_polygons"], result["room_types"])):
        cls   = int(t.get("class", 0))
        label, color = _label_and_color("room", cls)

        # Handle different Shapely geometry types (Polygon, MultiPolygon, GeometryCollection)
        if poly.geom_type == 'Polygon':
            polys = [poly]
        elif hasattr(poly, 'geoms'):
            polys = [g for g in poly.geoms if g.geom_type == 'Polygon']
            if not polys:
                # If no polygons, try to find them in nested multi-geometries
                polys = []
                for g in poly.geoms:
                    if hasattr(g, 'geoms'):
                        polys.extend([sub_g for sub_g in g.geoms if sub_g.geom_type == 'Polygon'])
        else:
            polys = []

        for p_idx, p in enumerate(polys):
            coords = [
                [float(x), float(y)]
                for x, y in list(p.exterior.coords)[:-1]   # drop closing duplicate
            ]
            rooms.append({
                "id":       idx + len(elements) + p_idx,
                "vertices": coords,
                "type":     "room",
                "class":    cls,
                "label":    label,
                "color":    color,
            })

    return {
        "elements":   elements,
        "rooms":      rooms,
        "orig_size":  [orig_w, orig_h],
        "resized_size": [resized_w, resized_h],
        "target_size": TARGET_SIZE,
    }


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static",
)
