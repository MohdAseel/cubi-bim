# Interactive Floorplan Viewer

An interactive web-based viewer for inspecting and correcting the model's detected floorplan elements — walls, rooms, and icons — with real-time overlay editing.

---

## Files

| File | Purpose |
|------|---------|
| `floortrans/export_schema.py` | Converts `get_polygons()` output to JSON |
| `server.py` | Flask server — runs inference and serves the viewer |
| `viewer/index.html` | Self-contained interactive web UI |

---

## Quick Start

### 1. Install Flask

```bash
pip install flask
```

### 2. Start the server

```bash
python server.py --weights model_best_val_loss_var.pkl
```

Then open **http://localhost:5000** in your browser.

To run the viewer without loading the model (for inspecting pre-saved JSON files only):

```bash
python server.py --no-model
```

### 3. Use the viewer

- **Load Image** — pick a floorplan PNG/JPG, or drag and drop it onto the canvas
- **Run Inference** — sends the image to the server, runs the model, and renders all detected elements as colored overlays
- **Load JSON** — load a previously saved schema file instead of running inference
- **Export JSON** — download the corrected schema after editing

---

## Server Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--weights` | `model_best_val_loss_var.pkl` | Path to trained model checkpoint |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `5000` | Port |
| `--no-model` | `False` | Start without loading the model (viewer-only) |

---

## JSON Schema

`export_schema.py` produces a portable JSON file describing all detected elements. You can save it, version it, pass it to other tools, or load it back into the viewer.

### Structure

```json
{
  "version": "1.0",
  "meta": {
    "width": 512,
    "height": 512,
    "image_path": null
  },
  "labels": {
    "room": { "0": "Background", "1": "Outdoor", "2": "Wall", "3": "Kitchen", "..." },
    "icon": { "0": "Empty", "1": "Window", "2": "Door", "..." },
    "wall": { "2": "Wall", "8": "Railing" }
  },
  "colors": {
    "room": { "3": "#fdb462", "4": "#8dd3c7", "..." },
    "icon": { "1": "#a0d4e8", "2": "#c8764a", "..." }
  },
  "elements": [
    {
      "id":         "wall-0",
      "type":       "wall",
      "class_idx":  2,
      "label":      "Wall",
      "color":      "#1a1a1a",
      "polygon":    [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
      "confidence": null
    },
    {
      "id":         "room-0",
      "type":       "room",
      "class_idx":  3,
      "label":      "Kitchen",
      "color":      "#fdb462",
      "polygon":    [[x,y], ...],
      "confidence": null
    },
    {
      "id":         "icon-0",
      "type":       "icon",
      "class_idx":  2,
      "label":      "Door",
      "color":      "#c8764a",
      "polygon":    [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
      "confidence": 0.8712
    }
  ]
}
```

### Element types

| `type` | Shape | `class_idx` range |
|--------|-------|-------------------|
| `wall` | 4-corner rectangle | 2 (Wall), 8 (Railing) |
| `room` | N-vertex polygon (Shapely) | 0–11 |
| `icon` | 4-corner rectangle | 0–10 |

### Room classes

| Index | Label |
|-------|-------|
| 0 | Background |
| 1 | Outdoor |
| 2 | Wall |
| 3 | Kitchen |
| 4 | LivingRoom |
| 5 | Bedroom |
| 6 | Bath |
| 7 | Entry |
| 8 | Railing |
| 9 | Storage |
| 10 | Garage |
| 11 | Room |

### Icon classes

| Index | Label |
|-------|-------|
| 0 | Empty |
| 1 | Window |
| 2 | Door |
| 3 | Closet |
| 4 | Appliance |
| 5 | Toilet |
| 6 | Sink |
| 7 | SaunaBench |
| 8 | Fireplace |
| 9 | Bathtub |
| 10 | Chimney |

---

## Using the Schema in Your Own Code

```python
from floortrans.post_prosessing import split_prediction, get_polygons
from floortrans.export_schema import polygons_to_schema, save_schema, load_schema

# After model inference
heatmaps, rooms, icons = split_prediction(output, (height, width), [21, 12, 11])
polygons, types, room_polygons, room_types = get_polygons(
    (heatmaps, rooms, icons), 0.4, [1, 2])

# Convert to schema and save
schema = polygons_to_schema(polygons, types, room_polygons, room_types, width, height,
                            image_path="path/to/floorplan.png")
save_schema(schema, "output/result.json")

# Load back later
schema = load_schema("output/result.json")
elements = schema["elements"]
walls = [e for e in elements if e["type"] == "wall"]
rooms = [e for e in elements if e["type"] == "room"]
icons = [e for e in elements if e["type"] == "icon"]
```

---

## Viewer Interactions

### Navigation

| Action | How |
|--------|-----|
| Zoom in / out | Mouse wheel |
| Pan | Click and drag on the background |
| Fit to window | **Fit** button |

### Editing

| Action | How |
|--------|-----|
| Select element | Click a polygon on the canvas, or click a row in the left list |
| Relabel class | Use the class dropdown in the right Properties panel |
| Resize wall / icon | Drag any of the 4 corner handles that appear on selection |
| Delete element | Press `Delete` / `Backspace`, or click the **Delete Selected** button |
| Deselect | Press `Escape` or click the background |
| Undo | `Ctrl+Z` or **↩ Undo** button |
| Redo | `Ctrl+Y` or **↪ Redo** button |

### Adding Elements

1. Choose the element type (`Wall`, `Room`, `Icon`) in the **Add** toolbar section
2. Choose the target class from the adjacent dropdown
3. Click **Draw Rect** to enter draw mode (button turns yellow)
4. Click and drag on the canvas to place the new rectangle
5. The element is created and immediately selected for further editing
6. Press `Escape` to exit draw mode without placing

### Filtering and Search

- Use the **All / Walls / Rooms / Icons** pills to filter the element list by type
- Type in the search box to filter by label name or element ID

### Exporting

| Button | Action |
|--------|--------|
| **Export JSON** | Downloads `floorplan_corrected.json` to your browser |
| **Save to Server** | POSTs the schema to `/export` on the running server and saves it to disk |

---

## Server API Reference

### `POST /infer`

Run model inference on an uploaded image.

**Request:** `multipart/form-data` with field `image` (PNG or JPG).

**Response:**
```json
{
  "schema":     { ... },
  "image_b64":  "<base64-encoded PNG>"
}
```

### `POST /export`

Save a (corrected) schema to disk on the server.

**Request body:**
```json
{
  "schema": { ... },
  "path":   "output/result.json"
}
```

**Response:**
```json
{ "saved": true, "path": "/absolute/path/to/result.json" }
```

### `POST /load`

Load an existing schema file from the server's filesystem.

**Request body:**
```json
{ "path": "output/result.json" }
```

**Response:**
```json
{ "schema": { ... } }
```

---

## Viewer-Only Mode (no server)

The viewer works entirely from local files without the server running. Open `viewer/index.html` directly in a browser and:

1. **Load Image** — pick the floorplan PNG from disk
2. **Load JSON** — pick a schema JSON saved from a previous run
3. Edit and **Export JSON** — the corrected file downloads locally

The **Run Inference** and **Save to Server** buttons require the Flask server to be running.
