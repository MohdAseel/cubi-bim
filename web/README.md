# CubiCasa Floorplan Editor — Web Application

An interactive browser-based editor for viewing and modifying the polygons produced by the CubiCasa5K inference pipeline.

---

## Overview

Upload a floorplan image → the backend runs the ML model → detected walls, icons, openings, and rooms are drawn as editable polygons on a canvas. Every property of every polygon can be changed in real time and the result exported as JSON.

```
┌────────────────────────────────┬──────────────────────┐
│                                │ PROPERTIES           │
│       Canvas                   │ Label ____________   │
│   (image + polygons)           │ Color  [■] Opacity   │
│   drag / resize / rotate       │ X ___  Y ___         │
│   wheel to zoom                │ W ___  H ___  Rot°   │
│   Alt+drag to pan              │ Vertices [x][y] …    │
│   Shift+click → distance       │ [Export JSON]        │
└────────────────────────────────┴──────────────────────┘
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| PyTorch | 2.x (CPU or CUDA) |
| Model weights | `model_best_val_loss_var.pkl` in the project root |

The ML dependencies from the parent project's `requirements.txt` must already be installed.

---

## Installation

```bash
# From the project root
cd web
pip install -r requirements.txt
```

`requirements.txt` adds three packages on top of the ML stack:

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API and static file serving |
| `uvicorn[standard]` | ASGI server |
| `python-multipart` | Multipart form upload support |

---

## Running the server

```bash
# From inside the web/ directory
python run.py
```

Or equivalently:

```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser. The model is loaded once at startup; subsequent requests reuse the same weights.

---

## Project structure

```
web/
├── backend.py          # FastAPI application — /predict endpoint + static serving
├── run.py              # One-line uvicorn launcher
├── requirements.txt    # Web-only dependencies
├── README.md           # This file
└── frontend/
    ├── index.html      # Single-page application shell
    ├── app.js          # All canvas and panel logic (Fabric.js)
    └── style.css       # Dark-theme UI styles
```

---

## API

### `POST /predict`

Accepts a multipart form upload and returns polygon data.

**Request**

```
Content-Type: multipart/form-data
Field: image  (any common raster format: PNG, JPEG, TIFF, …)
```

**Response**

```json
{
  "elements": [
    {
      "id": 0,
      "vertices": [[x, y], [x, y], [x, y], [x, y]],
      "type": "wall | icon",
      "class": 2,
      "label": "Wall",
      "color": "#C0392B"
    }
  ],
  "rooms": [
    {
      "id": 42,
      "vertices": [[x, y], ...],
      "type": "room",
      "class": 4,
      "label": "Living Room",
      "color": "#87CEEB"
    }
  ],
  "orig_size":    [W, H],
  "resized_size": [rW, rH],
  "target_size":  512
}
```

All `vertices` coordinates are in the **512 × 512 model space** (x = column, y = row). The frontend scales them to the canvas using the image's original aspect ratio.

### `GET /`

Serves `frontend/index.html`.

### `GET /static/{file}`

Serves files from `frontend/` (JS, CSS).

---

## Editor features

### Canvas interactions

| Action | How |
|--------|-----|
| Select polygon | Click |
| Move polygon | Drag |
| Resize polygon | Drag a corner handle |
| Rotate polygon | Drag the rotation handle (above bounding box) |
| Zoom | Scroll wheel |
| Pan | Alt + drag |
| Measure distance | Shift+click two polygons |

### Properties panel

All values are shown in **model-space pixels** (the 512 × 512 inference coordinate system).

| Section | Editable fields |
|---------|----------------|
| **Label** | Free-text name for the polygon |
| **Appearance** | Fill color (color picker), fill opacity (0–100 %), stroke width |
| **Transform** | Bounding-box origin (X, Y), Width, Height, Rotation (degrees) |
| **Vertices** | Each individual `[x, y]` vertex; add or remove vertices |
| **Metadata** | Type string, class index |

### Toolbar

| Control | Action |
|---------|--------|
| Upload Image | Load a new floorplan (triggers inference) |
| + / − | Zoom in / out |
| Fit | Reset viewport to the image |
| 100% | Reset zoom to 1× |
| Undo | Revert the last edit (up to 50 steps) |
| Elements ☑ | Show / hide wall, icon, and opening polygons |
| Rooms ☑ | Show / hide room fill polygons |
| Image ☑ | Show / hide the background image |
| Export JSON | Download all current polygon data as `.json` |

### Undo

Every edit that changes polygon geometry pushes a snapshot to a 50-step history. Clicking **Undo** restores the previous state, including any polygons that were deleted.

### Export

Clicking **Export JSON** downloads a file in this format:

```json
{
  "elements": [{ "id": 0, "label": "Wall", "type": "wall", "class": 2,
                 "color": "#C0392B", "vertices": [[x,y], ...] }],
  "rooms":    [{ "id": 10, "label": "Kitchen", "type": "room", "class": 3,
                 "color": "#FFD700", "vertices": [[x,y], ...] }]
}
```

---

## Class reference

### Room classes (type = `"room"` or `"wall"`)

| Index | Label |
|-------|-------|
| 0 | Background |
| 1 | Outdoor |
| 2 | Wall |
| 3 | Kitchen |
| 4 | Living Room |
| 5 | Bedroom |
| 6 | Bathroom |
| 7 | Entry / Corridor |
| 8 | Railing |
| 9 | Storage |
| 10 | Garage |
| 11 | Undefined |

### Icon classes (type = `"icon"`)

| Index | Label |
|-------|-------|
| 0 | (empty) |
| 1 | Window |
| 2 | Door |
| 3 | Closet |
| 4 | Electrical Appliance |
| 5 | Toilet |
| 6 | Sink |
| 7 | Sauna Bench |
| 8 | Fireplace |
| 9 | Bathtub |
| 10 | Chimney |

---

## Coordinate system

The model resizes the longest side of the input image to 512 pixels and zero-pads the shorter side, so all polygon coordinates live in a **512 × 512** grid. The frontend converts between this space and the canvas display on the fly:

```
canvas_x = image_offset_x + model_x × scale
canvas_y = image_offset_y + model_y × scale

scale = (display_width / original_width) × (original_longest_side / 512)
```

Properties panel fields always show model-space values; the canvas renders in display-space.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Server starts but `/predict` returns 503 | Model file `model_best_val_loss_var.pkl` not found in the project root. Download it and place it there. |
| Polygons appear offset from the image | The image may have embedded EXIF rotation. Pre-rotate the image before uploading, or strip EXIF metadata. |
| Slow inference | CUDA is not available. On CPU, TTA (4 rotations) takes 15–60 s depending on hardware. Pass `use_tta=False` in `predict_floorplan` for a 4× speedup at a small accuracy cost. |
| `python-multipart` not found | Run `pip install -r requirements.txt` from inside the `web/` directory. |
