"""
Convert get_polygons() output to a JSON schema for the web viewer.

Usage:
    from floortrans.post_prosessing import get_polygons, split_prediction
    from floortrans.export_schema import polygons_to_schema, save_schema

    heatmaps, rooms, icons = split_prediction(output, (H, W), [21, 12, 11])
    polygons, types, room_polygons, room_types = get_polygons(
        (heatmaps, rooms, icons), 0.4, [1, 2])
    schema = polygons_to_schema(polygons, types, room_polygons, room_types, W, H)
    save_schema(schema, "output.json")
"""

import json
import numpy as np

ROOM_LABELS = {
    0:  "Background",
    1:  "Outdoor",
    2:  "Wall",
    3:  "Kitchen",
    4:  "LivingRoom",
    5:  "Bedroom",
    6:  "Bath",
    7:  "Entry",
    8:  "Railing",
    9:  "Storage",
    10: "Garage",
    11: "Room",
}

ICON_LABELS = {
    0:  "Empty",
    1:  "Window",
    2:  "Door",
    3:  "Closet",
    4:  "Appliance",
    5:  "Toilet",
    6:  "Sink",
    7:  "SaunaBench",
    8:  "Fireplace",
    9:  "Bathtub",
    10: "Chimney",
}

WALL_LABELS = {2: "Wall", 8: "Railing"}

# Hex colours matching plotting.py's discrete_cmap
ROOM_COLORS = {
    0:  "#DCDCDC",  # Background
    1:  "#DCDCDC",  # Outdoor
    2:  "#1a1a1a",  # Wall
    3:  "#fdb462",  # Kitchen
    4:  "#8dd3c7",  # LivingRoom
    5:  "#b3de69",  # Bedroom
    6:  "#fccde5",  # Bath
    7:  "#80b1d3",  # Entry
    8:  "#808080",  # Railing
    9:  "#696969",  # Storage
    10: "#577a4d",  # Garage
    11: "#ffffb3",  # Room
}

ICON_COLORS = {
    0:  "transparent",
    1:  "#a0d4e8",  # Window
    2:  "#c8764a",  # Door
    3:  "#d4c5a9",  # Closet
    4:  "#ffcc44",  # Appliance
    5:  "#e8c4e0",  # Toilet
    6:  "#a4c8dc",  # Sink
    7:  "#c8a478",  # SaunaBench
    8:  "#e85c2c",  # Fireplace
    9:  "#dce8f0",  # Bathtub
    10: "#888888",  # Chimney
}


def _make_element(elem_type, class_idx, polygon_pts, confidence=None, elem_id=None):
    if elem_type == "wall":
        label = WALL_LABELS.get(class_idx, ROOM_LABELS.get(class_idx, f"class_{class_idx}"))
        color = ROOM_COLORS.get(class_idx, "#888888")
    elif elem_type == "room":
        label = ROOM_LABELS.get(class_idx, f"class_{class_idx}")
        color = ROOM_COLORS.get(class_idx, "#888888")
    else:  # icon
        label = ICON_LABELS.get(class_idx, f"class_{class_idx}")
        color = ICON_COLORS.get(class_idx, "#888888")

    return {
        "id":         elem_id,
        "type":       elem_type,
        "class_idx":  int(class_idx),
        "label":      label,
        "color":      color,
        "polygon":    polygon_pts,          # [[x, y], ...]
        "confidence": round(float(confidence), 4) if confidence is not None else None,
    }


def polygons_to_schema(polygons, types, room_polygons, room_types,
                       width, height, image_path=None):
    """
    Parameters
    ----------
    polygons       : np.ndarray (N, 4, 2)  – walls + icons + openings
    types          : list[dict]            – matching type/class/prob dicts
    room_polygons  : list[shapely.Polygon]
    room_types     : list[dict]
    width, height  : int                   – image pixel dimensions
    image_path     : str | None            – stored for reference only

    Returns
    -------
    dict  (JSON-serialisable)
    """
    elements = []
    counters = {"wall": 0, "room": 0, "icon": 0}

    # ── walls and icons (fixed 4-corner rectangles) ──────────────────────────
    for pol, t in zip(polygons, types):
        etype = t["type"]  # "wall" or "icon"
        cidx  = int(t["class"])
        conf  = t.get("prob")
        pts   = [[int(p[0]), int(p[1])] for p in pol]  # [[x,y], x4]
        eid   = f"{etype}-{counters[etype]}"
        counters[etype] += 1
        elements.append(_make_element(etype, cidx, pts, conf, eid))

    # ── room polygons (Shapely, variable vertex count) ────────────────────────
    for pol, t in zip(room_polygons, room_types):
        cidx = int(t["class"])
        # exterior.coords closes the ring (last == first); drop the duplicate
        coords = list(pol.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        pts  = [[float(x), float(y)] for x, y in coords]
        eid  = f"room-{counters['room']}"
        counters["room"] += 1
        elements.append(_make_element("room", cidx, pts, None, eid))

    return {
        "version": "1.0",
        "meta": {
            "width":      int(width),
            "height":     int(height),
            "image_path": image_path,
        },
        "labels": {
            "room": {str(k): v for k, v in ROOM_LABELS.items()},
            "icon": {str(k): v for k, v in ICON_LABELS.items()},
            "wall": {str(k): v for k, v in WALL_LABELS.items()},
        },
        "colors": {
            "room": {str(k): v for k, v in ROOM_COLORS.items()},
            "icon": {str(k): v for k, v in ICON_COLORS.items()},
        },
        "elements": elements,
    }


def save_schema(schema, path):
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)


def load_schema(path):
    with open(path) as f:
        return json.load(f)
